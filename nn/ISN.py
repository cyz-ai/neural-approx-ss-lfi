import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd
import numpy as np
import math
import time
from copy import deepcopy


class ISN(nn.Module):
    """ 
        Information-theoretic Statistics Network
    """
    def __init__(self, architecture, dim_y, hyperparams):
        super().__init__()

        # default hyperparameters
        self.estimator = 'JSD' if not hasattr(hyperparams, 'estimator') else hyperparams.estimator  
        self.bs = 200 if not hasattr(hyperparams, 'bs') else hyperparams.bs 
        self.lr = 1e-4 if not hasattr(hyperparams, 'lr') else hyperparams.lr
        self.wd = 0e-5 if not hasattr(hyperparams, 'wd') else hyperparams.wd
        self.n_neg = 300 if not hasattr(hyperparams, 'n_neg') else hyperparams.n_neg

        self.encode_layer = EncodeLayer(architecture, hyperparams)
        self.encode2_layer = EncodeLayer([dim_y] + architecture[1:-1] + [dim_y], None)
        self.critic_layer = CriticLayer(dim_x=architecture[-1], dim_y=dim_y, dim_hidden=200, n_layer=0, hyperparams=hyperparams)
    
    def encode(self, x):
        # s = s(x), get the summary statistic of x
        return self.encode_layer(x)
    
    def encode2(self, y):
        # theta = h(y), get the representation of y
        return self.encode2_layer(y)
        
    def MI(self, z, y, n=10):
        # [A]. Donsker-Varadhan Representation (MINE, ICML'18)
        if self.estimator == 'DV':
            m, d = z.size()
            z, y = self.encode(z), self.encode2(y)
            idx_pos = []
            idx_neg = []
            for i in range(n):
                idx_pos = idx_pos + np.linspace(0, m-1, m).tolist()
                idx_neg = idx_neg + torch.randperm(m).cpu().numpy().tolist()
            f_pos = self.critic_layer(z, y)
            f_neg = self.critic_layer(z[idx_pos], y[idx_neg])
            mi = f_pos.mean() - f_neg.exp().mean().log()
        # [B]. NWJ estimator (f-GAN, NIPS'17)
        if self.estimator == 'NWJ':
            m, d = z.size()
            z, y = self.encode(z), self.encode2(y)
            idx_pos = []
            idx_neg = []        
            for i in range(n):
                idx_pos = idx_pos + np.linspace(0, m-1, m).tolist()
                idx_neg = idx_neg + torch.randperm(m).cpu().numpy().tolist()
            f_pos = self.critic_layer(z, y)
            f_neg = self.critic_layer(z[idx_pos], y[idx_neg])
            mi = f_pos.mean() - (f_neg-1).exp().mean()
        # [C]. Jensen-shannon divergence (DeepInfoMax, ICLR'19)
        if self.estimator == 'JSD':
            m, d = z.size()
            z, y = self.encode(z), self.encode2(y)
            idx_pos = []
            idx_neg = []
            for i in range(n): 
                idx_pos = idx_pos + np.linspace(0, m-1, m).tolist()
                idx_neg = idx_neg + torch.randperm(m).cpu().numpy().tolist()
            f_pos = self.critic_layer(z, y)
            f_neg = self.critic_layer(z[idx_pos], y[idx_neg])
            A, B = -F.softplus(-f_pos), F.softplus(f_neg)
            mi = A.mean() - B.mean()
       # [D]. InfoNCE (InfoNCE, NIPS'18) 
        if self.estimator == 'InfoNCE':
            m, d = z.size()
            z, y = self.encode(z), self.encode2(y)
            idx_pos = []
            idx_neg = []
            for i in range(m):
                idx_pos = idx_pos + (np.zeros(m-1)+i).tolist()
                idx = torch.tensor(np.linspace(0, m-1, m))
                idx_neg = idx_neg + idx[idx.ne(i).nonzero().view(-1)].numpy().tolist()
            f_pos = self.critic_layer(z, y).exp() + 1e-10
            f_neg = self.critic_layer(z[idx_pos], y[idx_neg]).exp().view(m, m-1).sum(dim=1) + 1e-10
            A, B = f_pos.log(), f_neg.log()
            mi = A.mean() - B.mean()
        # [E]. Wasserstein dependency measure (WPC, NIPS'19)
        if self.estimator == 'WD':
            z, y = self.encode(z), y
            m, d, K = z.size()[0], z.size()[1], y.size()[1]
            z_shuffle = torch.zeros(n*m, d).to(z.device)
            y_shuffle = torch.zeros(n*m, K).to(z.device)   
            idx_pos = []
            idx_neg = []
            idx_neg2 = []
            for i in range(n):
                idx_pos = idx_pos + np.linspace(0, m-1, m).tolist()
                idx_neg = idx_neg + torch.randperm(m).cpu().numpy().tolist()
                idx_neg2 = idx_neg2 + torch.randperm(m).cpu().numpy().tolist()
            alpha = torch.rand(m*n, 1).to(z.device)
            z_mix = alpha*z[idx_pos] + (1-alpha)*z[idx_neg]
            y_mix = alpha*y[idx_neg] + (1-alpha)*y[idx_neg2]
            zy_mix = autograd.Variable(torch.cat((z_mix, y_mix), dim=1), requires_grad=True)
            f_pos = self.critic_layer(z[idx_pos], self.encode2(y[idx_pos]))
            f_neg = self.critic_layer(z[idx_neg], self.encode2(y[idx_neg2])) 
            f_mix = self.critic_layer(zy_mix[:,0:d], self.encode2(zy_mix[:,d:]))
            gradients = autograd.grad(outputs=f_mix, inputs=zy_mix,
                                      grad_outputs=torch.ones(f_mix.size(), device=z.device),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
            gp = F.relu(gradients.norm(2, dim=1)-1)
            mi = f_pos.mean() - f_neg.mean() - self.training*1e2*gp.mean()
        # [F]. Distance correlation (Annals of Statistics'07)
        if self.estimator == 'DC':
            m, d = z.size()
            z, y = self.encode(z), y
            A = torch.cdist(z, z, p=2)
            B = torch.cdist(y, y, p=2)
            A_row_sum, A_col_sum = A.sum(dim=0, keepdim=True), A.sum(dim=1, keepdim=True)
            B_row_sum, B_col_sum = B.sum(dim=0, keepdim=True), B.sum(dim=1, keepdim=True)
            a = A - A_row_sum/(m-2) - A_col_sum/(m-2) + A.sum()/((m-1)*(m-2))
            b = B - B_row_sum/(m-2) - B_col_sum/(m-2) + B.sum()/((m-1)*(m-2))
            AB, AA, BB = (a*b).sum()/(m*(m-3)), (a*a).sum()/(m*(m-3)), (b*b).sum()/(m*(m-3))
            mi = AB**0.5/(AA**0.5 * BB**0.5)**0.5
        return mi
    
    def LOSS(self, x, y, n=10):
        return -self.MI(x, y, n)
    
    def learn(self, x, y):    
        # hyperparams 
        T = 3000
        T_NO_IMPROVE_THRESHOLD = 200
        
        # divide train & val 
        n = len(x)
        n_train = int(0.80*n)
        bs = self.bs if n_train>self.bs else n_train
        idx = torch.randperm(n) 
        x_train, y_train = x[idx[0:n_train]], y[idx[0:n_train]]
        x_val, y_val = x[idx[n_train:n]], y[idx[n_train:n]]
        print('validation size=', n_train/n)
        
        # learn in loops
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, weight_decay=self.wd)
        n_batch, n_val_batch = int(len(x_train)/bs), int(len(x_val)/bs)+1
        best_val_loss, best_model_state_dict, no_improvement = math.inf, None, 0
        
        for t in range(T):
            # shuffle the batch
            idx = torch.randperm(len(x_train)) 
            x_train, y_train = x_train[idx], y_train[idx]
            x_chunks, y_chunks = torch.chunk(x_train, n_batch), torch.chunk(y_train, n_batch)
            x_v_chunks, y_v_chunks = torch.chunk(x_val, n_val_batch), torch.chunk(y_val, n_val_batch)

            # gradient descend
            self.train()
            start = time.time()
            for i in range(len(x_chunks)):
                optimizer.zero_grad()
                loss = -self.MI(x_chunks[i], y_chunks[i], n=self.n_neg)
                loss.backward()
                optimizer.step()
            end = time.time()

            # early stopping if val loss does not improve after some epochs
            self.eval()
            loss_val = torch.zeros(1, device=x.device)
            for j in range(len(x_v_chunks)):
                loss_val += -self.MI(x_v_chunks[j], y_v_chunks[j], n=self.n_neg)/len(x_v_chunks)
            improved = loss_val.item() < best_val_loss
            no_improvement = 0 if improved else no_improvement + 1
            best_val_loss = loss_val.item() if improved else best_val_loss     
            best_model_state_dict = deepcopy(self.state_dict()) if improved else best_model_state_dict
            if no_improvement >= T_NO_IMPROVE_THRESHOLD: break

            # report
            if t%50 == 0: print('finished: t=', t, 'loss=', loss.item(), 'loss val=', loss_val.item(), 'time=', (end-start)/len(x_chunks))

        # return the best snapshot in the history
        self.load_state_dict(best_model_state_dict)
        print('best val loss=', best_val_loss)
        return best_val_loss

    
    
    
        
class CriticLayer(nn.Module): 
    def __init__(self, dim_x, dim_y, dim_hidden, n_layer=1, hyperparams=None):
        super().__init__()        
        self.fc = nn.Linear(dim_x + dim_y, dim_hidden)
        self.drop = nn.Dropout(p=0.20)
        self.main = nn.Sequential(
            *(nn.Linear(dim_hidden, dim_hidden, bias=True) for i in range(n_layer)),
        )
        self.out = nn.Linear(dim_hidden, 1)
        self.dropout = False if not hasattr(hyperparams, 'dropout') else hyperparams.dropout 
        self.clip = 999999 if not hasattr(hyperparams, 'clip') else hyperparams.clip 
        
    def forward(self, x, y):
        xy = torch.cat((x,y), dim=1)
        h = self.fc(xy)
        h = F.relu(h)
        h = self.drop(h) if self.dropout else h
        for layer in self.main:
            h = F.relu(layer(h))
        out = self.out(h)
        out = torch.clamp(out, -self.clip, self.clip)
        return out
    
    
    
    
        
class EncodeLayer(nn.Module):
    def __init__(self, architecture, hyperparams=None):
        super().__init__()
        self.type = 'plain' if not hasattr(hyperparams, 'type') else hyperparams.type 
        self.dropout = False if not hasattr(hyperparams, 'dropout') else hyperparams.dropout 
        self.main = nn.Sequential( 
           *(nn.Linear(architecture[i+1], architecture[i+2], bias=True) for i in range(len(architecture)-3)),
        )      
        if self.type == 'plain':
            self.plain = nn.Linear(architecture[0], architecture[1], bias=True)
        if self.type == 'lstm':
            self.front = nn.LSTM(1, architecture[1], num_layers=1)
        if self.type == 'cnn':
            self.cnn = nn.Sequential(
                 nn.Conv1d(in_channels=1, out_channels=50, kernel_size=5, stride=1),
                 nn.ReLU(),
                 nn.Flatten(),
                 nn.Linear((architecture[0]-4*1)*50, architecture[1])   # d = D - (K-1)L
            )
        if self.type == 'cnn2d':
            self.cnn2d = nn.Sequential(
                 nn.Conv2d(in_channels=1, out_channels=50, kernel_size=2, stride=1),
                 nn.ReLU(),
                 nn.Flatten(),
                 nn.Linear(50*(int(architecture[0]**0.5)-1)**2, architecture[1])# d = D - (K-1)L
            )
        self.drop = nn.Dropout(p=0.20)
        self.out = nn.Linear(architecture[-2], architecture[-1], bias=True)
        self.N_layers = len(architecture) - 1
        self.architecture = architecture
            
    def front_end(self, x):
        if self.type == 'lstm':
            n, d = x.size()
            x = x.view(n, d, 1).permute(1, 0, 2)
            lstm_hidden_size = self.architecture[0]
            lstm_num_layers = 1
            hidden_cell = (torch.zeros(lstm_num_layers,n,lstm_hidden_size).to(x.device), 
                           torch.zeros(lstm_num_layers,n,lstm_hidden_size).to(x.device))
            out, _ = self.lstm(x, hidden_cell)
            x = out[-1]
        if self.type == 'cnn':
            n, d = x.size()
            x = x.view(n, 1, d)
            x = self.cnn(x)
        if self.type == 'cnn2d':
            n, d = x.size()
            x = x.view(n, 1, int(d**0.5), int(d**0.5))
            x = self.cnn2d(x)
        if self.type == 'plain':
            x = self.plain(x) 
        return x
        
    def forward(self, x):
        x = self.front_end(x)
        for layer in self.main: x = F.relu(layer(x))
        x = self.drop(x) if self.dropout else x
        x = self.out(x)
        return x
