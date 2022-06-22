import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd
import numpy as np
import scipy
import math
import time
import optimizer
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
        self.lr = 5e-4 if not hasattr(hyperparams, 'lr') else hyperparams.lr
        self.wd = 0e-5 if not hasattr(hyperparams, 'wd') else hyperparams.wd
        self.n_neg = 20 if not hasattr(hyperparams, 'n_neg') else hyperparams.n_neg
        self.encode_y = True if not hasattr(hyperparams, 'encode_y') else hyperparams.encode_y

        self.encode_layer = EncodeLayer(architecture, dim_y, hyperparams)
        self.encode2_layer = EncodeLayer([dim_y] + architecture[1:], dim_y, None)
        self.critic_layer = CriticLayer(architecture, architecture[-1], hyperparams)
        
        self.encodeb_layer = EncodeLayer(architecture[0:-1] + [1], 1, hyperparams)
        self.encode2b_layer = EncodeLayer([dim_y] + architecture[1:-1] + [1], 1, None)
    
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
        # [B]. NWJ estimator (f-div, NIPS'17)
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
            z, y = self.encode(z), self.encode2(y) if self.encode_y else y
            idx_pos = []
            idx_neg = []
            for i in range(n): 
                idx_pos = idx_pos + np.linspace(0, m-1, m).tolist()
                idx_neg = idx_neg + torch.randperm(m).cpu().numpy().tolist()
            f_pos = self.critic_layer(z, y)
            f_neg = self.critic_layer(z[idx_pos], y[idx_neg])
            A, B = -F.softplus(-f_pos), F.softplus(f_neg)
            mi = A.mean() - B.mean()
        # [C]. Jensen-shannon divergence (DeepInfoMax, ICLR'19)
        if self.estimator == 'JSD2':
            m, d = z.size()
            z, zz, y, yy = self.encode(z), self.encodeb_layer(z), self.encode2(y), self.encode2b_layer(y)
            idx_pos = []
            idx_neg = []
            for i in range(n): 
                idx_pos = idx_pos + np.linspace(0, m-1, m).tolist()
                idx_neg = idx_neg + torch.randperm(m).cpu().numpy().tolist()
            def fake_critic(z, zz, y, yy): return (z*y).sum(dim=1) + yy.view(-1) + zz.view(-1)  
            f_pos = fake_critic(z[idx_pos], zz[idx_pos], y[idx_pos], yy[idx_pos])
            f_neg = fake_critic(z[idx_pos], zz[idx_pos], y[idx_neg], yy[idx_neg])
            A, B = -F.softplus(-f_pos), F.softplus(f_neg)
            mi = A.mean() - B.mean()
        # [E]. Wasserstein dependency measure (WPC, NIPS'19)
        if self.estimator == 'WD':
            z, y = self.encode(z), self.encode2(y)
            m, d, K = z.size()[0], z.size()[1], y.size()[1] 
            idx_pos = []
            idx_neg = []
            for i in range(n):
                idx_pos = idx_pos + np.linspace(0, m-1, m).tolist()
                idx_neg = idx_neg + torch.randperm(m).cpu().numpy().tolist()
            f_pos = self.critic_layer(z, y)
            f_neg = self.critic_layer(z[idx_pos], y[idx_neg])
            mi = f_pos.mean() - f_neg.mean()
        # [F]. Distance correlation (Annals of Statistics'07)
        if self.estimator == 'DC':
            m, d = z.size()
            z, y = self.encode(z), self.encode2(y) if self.encode_y else y
            A = torch.cdist(z, z, p=2)
            B = torch.cdist(y, y, p=2)
            A_row_sum, A_col_sum = A.sum(dim=0, keepdim=True), A.sum(dim=1, keepdim=True)
            B_row_sum, B_col_sum = B.sum(dim=0, keepdim=True), B.sum(dim=1, keepdim=True)
            a = A - A_row_sum/(m-2) - A_col_sum/(m-2) + A.sum()/((m-1)*(m-2))
            b = B - B_row_sum/(m-2) - B_col_sum/(m-2) + B.sum()/((m-1)*(m-2))
            AB, AA, BB = (a*b).sum()/(m*(m-3)), (a*a).sum()/(m*(m-3)), (b*b).sum()/(m*(m-3))
            mi = AB**0.5/(AA**0.5 * BB**0.5)**0.5
        # [G]. Renyi correlation
        if self.estimator == 'Renyi':
            m, d = z.size()
            z, y = self.bridge_layer(self.encode(z)), self.encode2(y) if self.encode_y else y
            z_mu, z_std = z.mean(dim=0, keepdim=True), z.std(dim=0, keepdim=True)
            z = (z-z_mu)/(z_std + 1e-10)
            y_mu, y_std = y.mean(dim=0, keepdim=True), y.std(dim=0, keepdim=True)
            y = (y-y_mu)/(y_std + 1e-10)
            yy = (y*z).mean(dim=0)
            corr = yy.abs()
            mi = corr.mean()
        # [H]. Reverse KL 
        if self.estimator == 'R-KL':
            m, d = z.size()
            z, y = self.encode(z), self.encode2(y)
            idx_pos = []
            idx_neg = []        
            for i in range(n):
                idx_pos = idx_pos + np.linspace(0, m-1, m).tolist()
                idx_neg = idx_neg + torch.randperm(m).cpu().numpy().tolist()
            f_pos = z*y
            f_neg = self.critic_layer(z[idx_pos], y[idx_neg])
            mi = f_pos.mean() - (f_pos-1).exp().mean()
        return mi
    
    def objective_func(self, x, y):
        return self.MI(x, y, n=self.n_neg)
    
    def learn(self, x, y):
        loss_value = optimizer.NNOptimizer.learn(self, x, y)
        return loss_value
    


    
            
class CriticLayer(nn.Module): 
    def __init__(self, architecture, dim_y, hyperparams=None):
        super().__init__()       
        dim_x, dim_y, dim_hidden = architecture[-1], dim_y, 200
        # WD case; need to do spectral normalization
        if hyperparams.estimator is 'WD':
            self.main = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(dim_x + dim_y, dim_hidden), n_power_iterations=5),
            )
            self.out = nn.utils.spectral_norm(nn.Linear(dim_hidden, 1), n_power_iterations=5)
            self.alpha = nn.Parameter(torch.Tensor(1))
        # Other cases; need to do noting
        else:
            self.main = nn.Sequential(
                nn.Linear(dim_x + dim_y, dim_hidden),
            )
            self.out = nn.Linear(dim_hidden, 1)

    def forward(self, x, y, others=None):
        h = torch.cat((x,y), dim=1)
        if others is None:
            h = self.main(h)
        else:
            h = self.main(h) + self.cond(others)
        h = torch.tanh(h)
        out = self.out(h)
        return out 
    
    
        
class EncodeLayer(nn.Module):
    def __init__(self, architecture, dim_y, hyperparams=None):
        super().__init__()
        self.type = 'plain' if not hasattr(hyperparams, 'type') or hyperparams is None else hyperparams.type 
        self.dropout = False if not hasattr(hyperparams, 'dropout') or hyperparams is None else hyperparams.dropout 
        self.main = nn.Sequential( 
           *(nn.Linear(architecture[i+1], architecture[i+2], bias=True) for i in range(len(architecture)-3)),
        )  
        if self.type == 'plain':
            self.plain = nn.Linear(architecture[0], architecture[1], bias=True)
        if self.type == 'iid':
            self.enn = nn.Sequential(
                 nn.Conv1d(in_channels=1, out_channels=50, kernel_size=1, stride=1),
                 nn.ReLU(),
                 nn.Conv1d(in_channels=50, out_channels=architecture[1], kernel_size=1, stride=1),
            )
        if self.type == 'cnn1d':
            self.cnn = nn.Sequential(
                 nn.Conv1d(in_channels=1, out_channels=50, kernel_size=3, stride=1),
                 nn.ReLU(),
                 nn.Conv1d(in_channels=50, out_channels=architecture[1], kernel_size=3, stride=1),
            )
        if self.type == 'cnn2d':
            self.cnn2d = nn.Sequential(
                 nn.Conv2d(in_channels=1, out_channels=50, kernel_size=2, stride=1),
                 nn.ReLU(),
                 nn.Flatten(),
                 nn.Linear(50*(int(architecture[0]**0.5)-1)**2, architecture[1])# d = D - (K-1)L
            )
        self.drop = nn.Dropout(p=0.20)
        self.out = nn.Sequential(
            nn.Linear(architecture[-2], architecture[-1], bias=True),
        )
        self.N_layers = len(architecture) - 1
        self.architecture = architecture
            
    def front_end(self, x):
        # i.i.d data
        if self.type == 'iid':
            n, d = x.size()
            x = x.view(n, 1, d)  # n*1*d
            x = self.enn(x)      # n*k*d
            x = x.sum(dim=2)     # n*k
        # Time-series data
        if self.type == 'cnn1d':
            n, d = x.size()
            x = x.view(n, 1, d)  # n*1*d
            x = self.cnn(x)      # n*k*d'
            x = x.sum(dim=2)     # n*k
        # Image data
        if self.type == 'cnn2d':
            n, d = x.size()
            x = x.view(n, 1, int(d**0.5), int(d**0.5))
            x = self.cnn2d(x)    # n*k
        # default
        if self.type == 'plain':
            x = self.plain(x) 
        return x
        
    def forward(self, x):
        x = self.front_end(x)
        for layer in self.main: x = F.relu(layer(x))
        x = self.drop(x) if self.dropout else x
        x = self.out(x)
        return x
    
    

class TransformLayer(nn.Module):
        
    def __init__(self, architecture, dim_y, hyperparams):
        super().__init__()     
        self.repeat = int(architecture[-1]/dim_y)
        self.A_ = nn.Parameter(torch.Tensor(1))
        self.B_ = nn.Parameter(torch.Tensor(1))
        self.g_ = nn.Parameter(torch.Tensor(1))
        self.k_ = nn.Parameter(torch.Tensor(1))
        self.A_.data.fill_(0)
        self.B_.data.fill_(0)  
        self.g_.data.fill_(0)
        self.k_.data = (torch.zeros(self.k_.size())-0.69315).data
     
    def reparam(self):
        A,B,g,k = self.A_,F.softplus(self.B_),self.g_,F.softplus(self.k_)-0.5
        return A,B,g,k
    
    def forward(self, y):
        n,d = y.size()
        t = y.clone().detach().cpu()
        Y = []
        for j in range(self.repeat):
            y2 = torch.zeros(n, d)
            for i in range(d):
                sample = np.atleast_1d(t[:,i])
                rank = scipy.stats.rankdata(sample).astype(float)
                cdf = (rank+1)/(n+2)
                z = scipy.stats.norm.ppf(cdf)
                A,B,g,k = self.reparam()
                y2[:,i] = self.Q(torch.Tensor(z), A,B,g,k)
            Y.append(y2)
        return torch.cat(Y, dim=1).to(y.device)
    
    def Q(self, z,A,B,g,k):
        # preparation
        c = 0.80
        g,k = torch.tensor(g).float(), torch.tensor(k).float()
        A,B = torch.tensor(A).float(), torch.tensor(B).float()
        # forward
        w = (1-torch.exp(-g*z))/(1+torch.exp(-g*z))
        v = z*(1+z*z).pow(k)
        x = A + B*(1 + c*w)*v
        return x
    

