import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
import torch.autograd as autograd
import time
import optimizer
from nn.ISN import EncodeLayer
from copy import deepcopy


class SSN(nn.Module):
    """ 
        Score-matching Statistic Network
    """
    def __init__(self, architecture, dim_y, hyperparams):
        super().__init__()
        self.bs = 400 if not hasattr(hyperparams, 'bs') else hyperparams.bs 
        self.lr = 1e-3 if not hasattr(hyperparams, 'lr') else hyperparams.lr
        self.wd = 0e-5 if not hasattr(hyperparams, 'wd') else hyperparams.wd
        self.type = 'plain' if not hasattr(hyperparams, 'type') else hyperparams.type 
        self.dropout = False if not hasattr(hyperparams, 'dropout') else hyperparams.dropout 
        
        self.score_layer = ScoreLayer(architecture[-1], dim_y, 100, n_layer=1)
        self.encode_layer = EncodeLayer(architecture, dim_y, hyperparams)

    def encode(self, x):
        # s = s(x), get the summary statistic of x
        return self.encode_layer(x)
        
    def SM(self, x, y):
        n, dim = y.size()
        x = x.clone().requires_grad_(True)
        y = y.clone().requires_grad_(True)
        log_energy = self.score_layer(x, y)                                    # log f(y|x), R^d                m*1
        score = autograd.grad(log_energy.sum(), y, create_graph=True)[0]       # d_y log f(y|x),                m*d
        loss1 = 0.5*torch.norm(score, dim=1) ** 2                              # |d_y|^2                        m*1
        loss2 = torch.zeros(n, device=x.device)                                # trace d_yy log f(y|x)          m*1
        for d in range(dim): 
            loss2 += autograd.grad(score[:, d].sum(), y, create_graph=True, retain_graph=True)[0][:, d] 
        loss = loss1 + loss2
        return loss.mean()
    
    def objective_func(self, x, y):
        loss = self.SM(self.encode(x), y)
        return -loss
        
    def learn(self, x, y):
        loss_value = optimizer.NNOptimizer.learn(self, x, y)
        return loss_value   

        
        
class ScoreLayer(nn.Module): 
    def __init__(self, dim_x, dim_y, dim_hidden, n_layer=1):
        super().__init__()        
        self.fc1 = nn.Linear(dim_x, dim_hidden)
        self.fc2 = nn.Linear(dim_y, dim_hidden)
        self.merge = nn.Linear(2*dim_hidden, dim_hidden)
        self.main = nn.Sequential(
            *(nn.Linear(dim_hidden, dim_hidden, bias=True) for i in range(n_layer)),
            #nn.Dropout(p=0.2)
        )
        self.out = nn.Linear(dim_hidden, 1)
        
    def forward(self, x, y):
        #h = self.fc1(x) + self.fc2(y)
        h = torch.cat([self.fc1(x), self.fc2(y)], dim=1)
        h = F.leaky_relu(self.merge(h), 0.2)
        for layer in self.main:
            h = F.leaky_relu(layer(h), 0.2)
        out = self.out(h)
        return out
    
    
    
