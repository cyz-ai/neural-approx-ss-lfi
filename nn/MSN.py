import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
from copy import deepcopy


class RSN(nn.Module):
    """ 
        Regression statistic network
    """
    def __init__(self, architecture, dim_y):
        super().__init__()
        self.out_type = 'linear'
        self.main = nn.Sequential(
            *(nn.Linear(architecture[i], architecture[i+1], bias=True) for i in range(len(architecture)-1)),
            #nn.Dropout(p=0.5)
        )
        self.out = nn.Linear(architecture[-1], dim_y, bias=True)
        
    def forward(self, x):
        for layer in self.main:
            x = F.relu(layer(x))
        y = self.out(x)
        if self.out_type == 'linear':
            y=y
        if self.out_type == 'log-exp':
            y=torch.log(y.exp()+1)
        if self.out_type == 'sigmoid':
            y=torch.sigmoid(y)
        return y

    def encode(self, x, l=1):
        i = 0
        for layer in self.main:
            x = F.relu(layer(x))
            i += 1
            if i == l: break
        return x
    
    def mse(self, x, y):
        J = torch.norm(x-y, dim=1)**2
        return J
            
    # x=data, y=thetas
    def learn(self, x, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4, weight_decay=0e-5)
        bs = 500
        T = 8000

        # divide train & val
        n = len(x)
        n_val = int(0.80*n)
        idx = torch.randperm(n)
        x_train, y_train = x[idx[0:n_val]], y[idx[0:n_val]]
        x_val, y_val = x[idx[n_val:n]], y[idx[n_val:n]]
        
        # learn!
        n_batch = int(len(x_train)/bs)
        best_val_loss, best_model_state_dict, no_improvement = math.inf, None, 0
        for t in range(T):
            # shuffle 
            idx = torch.randperm(len(x_train))
            x_train, y_train = x_train[idx], y_train[idx]
            x_chunks, y_chunks = torch.chunk(x_train, n_batch), torch.chunk(y_train, n_batch)
            
            # optimize
            for i in range(len(x_chunks)):
                optimizer.zero_grad()
                y_pred, y_target = self.forward(x_chunks[i]), y_chunks[i]
                loss = self.mse(y_pred, y_target).mean()
                loss.backward()
                optimizer.step()
                
            # early stopping if val loss does not improve after some epochs
            y_pred_val, y_target_val = self.forward(x_val), y_val
            loss_val = self.mse(y_pred_val, y_target_val).mean()
            improved = loss_val.item() < best_val_loss
            no_improvement = 0 if improved else no_improvement + 1
            best_val_loss = loss_val.item() if improved else best_val_loss     
            best_model_state_dict = deepcopy(self.state_dict()) if improved else best_model_state_dict
            if no_improvement >= 50: break
                
            # report
            if t%int(T/20) == 0: print('finished: t=', t, 'loss=', loss.item(), 'loss val=', loss_val.item())
        
        # return the best model
        self.load_state_dict(best_model_state_dict)
        print('best val loss=', best_val_loss)
        return best_val_loss