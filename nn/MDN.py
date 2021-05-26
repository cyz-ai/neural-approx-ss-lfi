import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions as distribution
import math
import numpy as np
import time
from copy import deepcopy


class MDN(nn.Module):
    """ 
        Mixture density network 
    """
    def __init__(self, n_in, n_hidden, n_out, K=1):
        super(MDN, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )
        self.K = K
        self.dim = n_out
        self.coeff_layer = CoeffLayer(n_hidden, K)
        self.mean_layer = nn.ModuleList([MeanLayer(n_hidden, n_out) for i in range(K)])
        self.cov_layer = nn.ModuleList([CovLayer(n_hidden, n_out) for i in range(K)])
        
    def forward(self, cond_inputs):
        # nn(x) = {coeff}, {mu}, {cov}
        h = self.main(cond_inputs)
        mu_array, C_array, log_det_array = [], [], []
        for k in range(self.K):
            # > mu
            mu = self.mean_layer[k](h)
            mu_array.append(mu)
            # > cov
            C, log_det = self.cov_layer[k](h)
            C_array.append(C)
            log_det_array.append(log_det)
        coeff = self.coeff_layer(h)
        return coeff, mu_array, C_array, log_det_array
    
    def sample(self, cond_inputs, n=1):
        device = cond_inputs.device
        coeff, mu_array, C_array, log_det_array = self.forward(cond_inputs)
        categorical = distribution.Categorical(coeff)
        samples = []
        for i in range(n):
            k = categorical.sample()    # pick a component
            mu, C = mu_array[k][0], C_array[k][0].inverse()
            V = C.mm(C.t())
            normal = distribution.MultivariateNormal(mu, V)
            x = normal.sample()        
            samples.append(x)
        return torch.cat(samples, dim=0)
            
    def log_probs(self, inputs, cond_inputs):
        # pdf = \sum coeff[k] * N(x; mu[k], cov[k])
        coeff, mu_array, C_array, log_det_array = self.forward(cond_inputs)
        prob = torch.zeros(len(inputs)).to(inputs.device)
        normal = distribution.Normal(torch.tensor([0.0]).to(inputs.device), torch.tensor([1.0]).to(inputs.device))
        for k in range(self.K):   # <- pdf for each Gaussian component
            mu, C, log_det = mu_array[k], C_array[k], log_det_array[k]
            z = (inputs - mu).unsqueeze(dim=1)
            C_T = C.transpose(dim0=1, dim1=2)
            z = z.bmm(C_T)
            z = z.squeeze(dim=1)
            log_base_prob = normal.log_prob(z).sum(dim=1)
            log_prob = log_base_prob + log_det
            prob += coeff[:,k] * log_prob.exp() 
        return (prob + 1e-12).log()
    
    def learn(self, inputs, cond_inputs, weights=None):
        # optimizer 
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4, weight_decay=1e-5)
        T = 5000

        # divide train & val
        x, y, w = inputs, cond_inputs, torch.zeros(len(inputs)).to(inputs.device)+1.0 if weights is None else weights.view(-1)
        n = len(x)
        n_train = int(0.80*n)
        bs = 1000 if n_train>1000 else n_train
        idx = torch.randperm(n)
        x_train, x_val =  x[idx[0:n_train]], x[idx[n_train:n]]
        y_train, y_val =  y[idx[0:n_train]], y[idx[n_train:n]]
        w_train, w_val =  w[idx[0:n_train]], w[idx[n_train:n]]
        
        # go
        N = int(len(x_train)/bs)
        best_val_loss, best_model_state_dict, no_improvement = math.inf, None, 0
        for t in range(T):
            # shuffle 
            idx = torch.randperm(len(x_train))
            x_train, y_train, w_train = x_train[idx], y_train[idx], w_train[idx]
            x_chunks, y_chunks, w_chunks = torch.chunk(x_train, N), torch.chunk(y_train, N), torch.chunk(w_train, N)
            
            # loss
            for i in range(len(x_chunks)):
                optimizer.zero_grad()
                loss = -(self.log_probs(inputs=x_chunks[i], cond_inputs=y_chunks[i])*w_chunks[i]).mean()
                loss.backward()
                optimizer.step()
                
            # early stopping if val loss does not improve after 100 epochs
            loss_val = -(self.log_probs(inputs=x_val, cond_inputs=y_val)*w_val).mean()
            no_improvement += 1
            if loss_val.item() < best_val_loss:
                no_improvement = 0 
                best_val_loss = loss_val.item()  
                best_model_state_dict = deepcopy(self.state_dict())
            if no_improvement >= 100: break
                
            # report
            if t%int(T/20) == 0: print('finished: t=', t, 'loss=', loss.item(), 'loss val=', loss_val.item())
        
        # return the best model
        self.load_state_dict(best_model_state_dict)
        print('best val loss=', best_val_loss)
        return loss.item()
        
class CoeffLayer(nn.Module):
    def __init__(self, n_in, K):
        super(CoeffLayer, self).__init__()
        self.n_in = n_in
        self.K = K
        self.linear = nn.Linear(n_in, K)
        
    def forward(self, h):
        m, d = h.size()
        out = self.linear(h)
        s = out.exp()
        coeff = s/s.sum(dim=1, keepdim=True) 
        return coeff
    
        
class MeanLayer(nn.Module): 
    def __init__(self, n_in, n_out):
        super(MeanLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        
    def forward(self, h):
        m, d = h.size()
        out = self.linear(h)
        mean = out.view(m, self.n_out)
        return mean
        

class CovLayer(nn.Module):
    def __init__(self, n_in, n_out):
        super(CovLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out*n_out)
   
    def mask(self, h):
        n = len(h)
        ones = 1 + torch.zeros(self.n_out, self.n_out)
        ltri_mask = torch.tril(ones, diagonal=-1).expand(n, self.n_out, self.n_out)
        diag_mask = torch.eye(self.n_out).expand(n, self.n_out, self.n_out)
        return ltri_mask.to(h.device), diag_mask.to(h.device)
        
    def forward(self, h):
        n, d = h.size()
        out = self.linear(h)
        out = out.view(n, self.n_out, self.n_out)
        ltri_mask, diag_mask = self.mask(h)
        ltri, diag = out*ltri_mask, (out.exp()*diag_mask)
        C = ltri + diag                    
        log_det = (out*diag_mask).sum(dim=2).sum(dim=1)
        return C, log_det   # x = C^{-1}z, z = Cx, det|C| = -det|dx/dz|  C^{-1}C^{-T} = Sigma
    
    
        
        
        
        
