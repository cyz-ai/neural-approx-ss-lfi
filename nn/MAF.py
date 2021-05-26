import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions as distribution
import math
import numpy as np
import time
from copy import deepcopy


class MAF(nn.Sequential):
    """ 
        Mask autoregressive flow
    """
    def __init__(self, n_blocks, n_inputs, n_hidden, n_cond_inputs):
        module = []
        self.n_blocks = n_blocks
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_cond_inputs = n_cond_inputs
        for _ in range(n_blocks): module += [MADE(n_inputs, n_hidden, n_cond_inputs), Reverse(n_inputs)]
        super().__init__(*module)
    
    def forward(self, inputs, cond_inputs=None, mode='direct'):
        self.num_inputs = inputs.size(-1)
        sum_logdet = torch.zeros(inputs.size(0), 1, device=inputs.device)
        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                sum_logdet += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                sum_logdet += logdet
        return inputs, sum_logdet
    
    def sample(self, num_samples=1, cond_inputs=None):
        noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples, _ = self.forward(noise, cond_inputs, mode='inverse')
        return samples
        
    def log_probs(self, inputs, cond_inputs):
        u, log_jacob = self.forward(inputs, cond_inputs)
        log_base_prob = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(dim=1, keepdim=True)
        return (log_base_prob + log_jacob).sum(dim=1)

    def learn(self, inputs, cond_inputs, weights=None):
        # optimizer 
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4, weight_decay=1e-5)
        T = 10000

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
                
            # early stopping if val loss does not improve after some epochs
            loss_val = -(self.log_probs(inputs=x_val, cond_inputs=y_val)*w_val).mean()
            no_improvement += 1
            if loss_val.item() < best_val_loss:
                no_improvement = 0 
                best_val_loss = loss_val.item()  
                best_model_state_dict = deepcopy(self.state_dict())
                
            if no_improvement >= 50: break
                
            # report
            if t%int(T/20) == 0: print('finished: t=', t, 'loss=', loss.item(), 'loss val=', loss_val.item())
        
        # return the best model
        self.load_state_dict(best_model_state_dict)
        print('best val loss=', best_val_loss)
        return loss.item()
    

class MADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """
    def __init__(self, num_inputs, num_hidden, num_cond_inputs=None):
        super(MADE, self).__init__()
        input_mask = self.get_mask(num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = self.get_mask(num_hidden, num_hidden, num_inputs, mask_type='hidden')
        output_mask = self.get_mask(num_hidden, num_inputs, num_inputs, mask_type='output')
        self.join = MaskedLinear(num_inputs, num_hidden, input_mask, num_cond_inputs)
        self.hiddens = nn.Sequential(nn.Tanh(),
                            MaskedLinear(num_hidden, num_hidden, hidden_mask), 
                            nn.Tanh())
        self.mu = MaskedLinear(num_hidden, num_inputs, output_mask)
        self.alpha = MaskedLinear(num_hidden, num_inputs, output_mask)
                            
    def get_mask(self, n_in, n_out, d, mask_type):
        if mask_type == 'input':
            in_degrees = torch.arange(n_in) 
            out_degrees = torch.arange(n_out) % (d-1)
            mask = (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float() 
        if mask_type == 'output':
            in_degrees = torch.arange(n_in) % (d-1)
            out_degrees = torch.arange(n_out)
            mask = (out_degrees.unsqueeze(-1) > in_degrees.unsqueeze(0)).float()
        if mask_type == 'hidden':
            in_degrees = torch.arange(n_in) % (d-1)
            out_degrees = torch.arange(n_out) % (d-1)
            mask = (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float() 
        return mask
        
    def forward(self, inputs, cond_inputs=None, mode='direct'):
        # x -> u, J
        if mode == 'direct':
            h = self.join(inputs, cond_inputs)
            h = self.hiddens(h)
            m, a = self.mu(h), self.alpha(h)
            u = (inputs - m)*torch.exp(-a)
            return u, -a.sum(dim=1, keepdim=True)
        else:
        # u -> x, J
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.join(x, cond_inputs)
                h = self.hiddens(h)
                m, a = self.mu(h), self.alpha(h)
                x[:, i_col] = inputs[:, i_col]*torch.exp(a[:, i_col])+m[:, i_col]
            return x, -a.sum(dim=1, keepdim=True)

        
class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask, cond_in_features=None):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(cond_in_features, out_features)
        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None):
        out = F.linear(inputs, self.linear.weight*self.mask, self.linear.bias)
        if cond_inputs is not None:
            out += self.cond_linear(cond_inputs)
        return out
    
    
class Reverse(nn.Module):
    """ An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """
    def __init__(self, num_inputs):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)