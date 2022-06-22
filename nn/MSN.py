import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
import time
import optimizer
from nn.ISN import EncodeLayer
from copy import deepcopy


class MSN(nn.Module):
    """ 
        Moment statistic network
    """
    def __init__(self, architecture, dim_y, hyperparams):
        super().__init__()
        self.bs = 400 if not hasattr(hyperparams, 'bs') else hyperparams.bs 
        self.lr = 1e-3 if not hasattr(hyperparams, 'lr') else hyperparams.lr
        self.wd = 0e-5 if not hasattr(hyperparams, 'wd') else hyperparams.wd
        self.type = 'plain' if not hasattr(hyperparams, 'type') else hyperparams.type 
        self.dropout = False if not hasattr(hyperparams, 'dropout') else hyperparams.dropout 
        
        self.encode_layer = EncodeLayer(architecture, dim_y, hyperparams)

    def forward(self, x):
        return self.encode(x)

    def encode(self, x):
        return self.encode_layer(x)
    
    def objective_func(self, x, y):
        yy = self.forward(x)
        J = torch.norm(yy-y, dim=1)**2
        return -J.mean()
    
    def learn(self, x, y):
        loss_value = optimizer.NNOptimizer.learn(self, x, y)
        return loss_value