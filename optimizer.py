import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd
import numpy as np
import math
import time
from copy import deepcopy



class NNOptimizer(nn.Module):
    
    @staticmethod 
    def divide_train_val(x, y):
        n = len(x)
        n_train = int(0.80*n)
        x_train, y_train = x[0:n_train], y[0:n_train]
        x_val, y_val = x[n_train:n], y[n_train:n]
        return  x_train, y_train, x_val, y_val
    
    @staticmethod
    def get_trunks(x, y, bs):
        n_batch = int(len(x)/bs) if len(x) > bs else 1
        x_chunks, y_chunks = torch.chunk(x, n_batch), torch.chunk(y, n_batch)
        return x_chunks, y_chunks
        
    @staticmethod 
    def learn(net, x, y):    
        # hyperparams 
        T = 2000 if not hasattr(net, 'max_iteration') else net.max_iteration
        PRINTING = True if not hasattr(net, 'trace_learning') else net.trace_learning   
        T_NO_IMPROVE_THRESHOLD = 800
        
        # divide train & val 
        n = len(x)
        x_train, y_train, x_val, y_val = NNOptimizer.divide_train_val(x, y)
        bs = net.bs if len(x_train)>net.bs else len(x_train)
        net.device = x.device
        
        # learn in loops
        #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=net.lr, weight_decay=net.wd)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=net.lr, weight_decay=net.wd)
        n_batch, n_val_batch = int(len(x_train)/bs), int(len(x_val)/bs) if len(x_val) > bs else 1
        best_val_loss, best_model_state_dict, no_improvement = math.inf, None, 0
                
        for t in range(T):
            # shuffle the batch
            idx = torch.randperm(len(x_train)) 
            x_train, y_train = x_train[idx], y_train[idx]
            x_chunks, y_chunks = torch.chunk(x_train, n_batch), torch.chunk(y_train, n_batch)
            x_v_chunks, y_v_chunks = torch.chunk(x_val, n_val_batch), torch.chunk(y_val, n_val_batch)

            # gradient descend
            net.train()
            for i in range(len(x_chunks)):
                optimizer.zero_grad()
                loss = -net.objective_func(x_chunks[i], y_chunks[i])
                if t>0:
                    loss.backward()
                    optimizer.step()
              
            # early stopping if val loss does not improve after some epochs
            net.eval()
            loss_val = torch.zeros(1, device=x.device)
            for j in range(len(x_v_chunks)):
                loss_val += -net.objective_func(x_v_chunks[j], y_v_chunks[j])/len(x_v_chunks)
            improved = loss_val.item() < best_val_loss
            no_improvement = 0 if improved else no_improvement + 1
            best_val_loss = loss_val.item() if improved else best_val_loss     
            best_model_state_dict = deepcopy(net.state_dict()) if improved else best_model_state_dict
            if no_improvement >= T_NO_IMPROVE_THRESHOLD: break

            # report
            if PRINTING and t%(T//10) == 0: 
               print('finished: t=', t, 'loss=', loss.item(), 'loss val=', loss_val.item(), 'best loss', best_val_loss)
                            
        # return the best snapshot in the history
        net.load_state_dict(best_model_state_dict)
        return best_val_loss

    
    
class NNAdvOptimizer(nn.Module):
    
    @staticmethod 
    def divide_train_val(x, y):
        n = len(x)
        n_train = int(0.80*n)
        x_train, y_train = x[0:n_train], y[0:n_train]
        x_val, y_val = x[n_train:n], y[n_train:n]
        return  x_train, y_train, x_val, y_val
    
    @staticmethod
    def rand_sample(x, y, n):
        idx = torch.randperm(len(x))
        xx, yy = x[idx[0:n]], y[idx[0:n]]
        return xx, yy
        
            
    @staticmethod 
    def learn(net, x, y):    
        # hyperparams 
        T = 750 if not hasattr(net.hyperparams, 'max_iteration') else net.hyperparams.max_iteration
        T_NO_IMPROVE_THRESHOLD = 200
        
        # divide train & val 
        n = len(x)
        n_train = int(0.80*n)
        bs = net.bs if n_train>net.bs else n_train
        x_train, y_train, x_val, y_val = NNAdvOptimizer.divide_train_val(x, y)
        print('validation size=', n_train/n)

        # optimizer
        optimizer = torch.optim.Adam(net.non_adv_module(), lr=net.lr, weight_decay=net.wd)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50)
        
        # intermediate results 
        n_batch, n_val_batch = int(len(x_train)/bs), int(len(x_val)/bs)+1
        best_val_obj, best_model_state_dict, no_improvement, best_t = -math.inf, None, 0, -9999
        
        for t in range(T):
            # shuffle the batch
            idx = torch.randperm(len(x_train)) 
            x_train, y_train = x_train[idx], y_train[idx]
            x_chunks, y_chunks = torch.chunk(x_train, n_batch), torch.chunk(y_train, n_batch)
            x_v_chunks, y_v_chunks = torch.chunk(x_val, n_val_batch), torch.chunk(y_val, n_val_batch)

            # max-step
            net.train()    
            t0 = time.time()
            net.train_adv_layer(x, y)
            t1 = time.time()
            adv_loss = net.objective_func(x_train, y_train, mode='redundancy')
            adv_loss_val = net.objective_func(x_val, y_val, mode='redundancy')
            suff_loss = net.objective_func(x_train, y_train, mode='sufficiency')
            suff_loss_val = net.objective_func(x_val, y_val, mode='sufficiency')
            loss_snapshot = [round(adv_loss.item(), 3), round(adv_loss_val.item(), 3)]
            loss_snapshot_suff = [round(suff_loss.item(), 3), round(suff_loss_val.item(), 3)]
 

            # early stopping if val loss does not improve after some epochs   
            net.eval()  
            loss_val = -net.objective_func(x_val, y_val)
            val_obj = -loss_val.item()
            improved = (val_obj - best_val_obj > 1e-4) 
            no_improvement = 0 if improved else no_improvement + 1
            best_t = t if improved else best_t
            best_val_obj = val_obj if improved else best_val_obj     
            best_model_state_dict = deepcopy(net.state_dict()) if improved else best_model_state_dict
            if no_improvement >= T_NO_IMPROVE_THRESHOLD and t >= 250: break

            # min-step
            net.train()
            for i in range(len(x_chunks)):
                optimizer.zero_grad()            
                loss = -net.objective_func(x_chunks[i], y_chunks[i])
                loss.backward()
                optimizer.step()
                            
            sched.step(loss_val)

            # report
            if t%(T//10) == 0: 
                print('t=', t, 'loss=', loss.item(), 'loss val=', loss_val.item(), 'adv loss=', adv_loss.item(), 'adv val=', adv_loss_val.item())
                print('loss_snapshot', loss_snapshot, 'loss_snapshot_suff', loss_snapshot_suff, 'best_obj_value', best_val_obj, 'time=', t1-t0)

        # return the best snapshot in the history
        net.load_state_dict(best_model_state_dict)
        print('best val loss=', best_val_obj, 't=', t, 'best_t', best_t)
        return best_val_obj
    
    
    
    
    
#     def learn(self, inputs, cond_inputs, weights=None):
#         # optimizer 
#         optimizer = torch.optim.Adam(self.parameters(), lr=5e-4, weight_decay=0e-5)
#         T = 10000

#         # divide train & val
#         x, y, w = inputs, cond_inputs, torch.zeros(len(inputs)).to(inputs.device)+1.0 if weights is None else weights.view(-1)
#         n = len(x)
#         n_train = int(0.80*n)
#         bs = 400 if n_train>1000 else n_train
#         idx = torch.randperm(n)
#         x_train, x_val =  x[idx[0:n_train]], x[idx[n_train:n]]
#         y_train, y_val =  y[idx[0:n_train]], y[idx[n_train:n]]
#         w_train, w_val =  w[idx[0:n_train]], w[idx[n_train:n]]
        
#         # go
#         N = int(len(x_train)/bs)
#         best_val_loss, best_model_state_dict, no_improvement = math.inf, None, 0
#         for t in range(T):
#             # shuffle 
#             idx = torch.randperm(len(x_train))
#             x_train, y_train, w_train = x_train[idx], y_train[idx], w_train[idx]
#             x_chunks, y_chunks, w_chunks = torch.chunk(x_train, N), torch.chunk(y_train, N), torch.chunk(w_train, N)
            
#             # loss
#             for i in range(len(x_chunks)):
#                 optimizer.zero_grad()
#                 loss = -(self.log_probs(inputs=x_chunks[i], cond_inputs=y_chunks[i])*w_chunks[i]).mean()
#                 loss.backward()
#                 optimizer.step()
                
#             # early stopping if val loss does not improve after some epochs
#             loss_val = -(self.log_probs(inputs=x_val, cond_inputs=y_val)*w_val).mean()
#             no_improvement += 1
#             if loss_val.item() < best_val_loss:
#                 no_improvement = 0 
#                 best_val_loss = loss_val.item()  
#                 best_model_state_dict = deepcopy(self.state_dict())
                
#             if no_improvement >= 50: break
                
#             # report
#             if t%int(T/20) == 0: print('finished: t=', t, 'loss=', loss.item(), 'loss val=', loss_val.item())
        
#         # return the best model
#         self.load_state_dict(best_model_state_dict)
#         print('best val loss=', best_val_loss)
#         return loss.item()