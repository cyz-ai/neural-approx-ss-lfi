import numpy as np
import torch
import torch.nn.functional as F 
import math
import scipy.stats as stats
from abc import ABCMeta, abstractmethod
import distributions 
import utils_math
from problems import ABC_problems
from nn import MAF
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
from copy import deepcopy


class GK_Problem(ABC_problems.ABC_Problem):

    '''
    The 2D g-and-k distribution
    '''

    def __init__(self, N=100, n=50):

        self.N = N                                                                       # number of posterior samples
        self.n = n                                                                       # length of the data vector x = {x_1, ..., x_n}

        self.prior = [distributions.uniform, distributions.uniform, distributions.uniform]
        self.prior_args =  np.array([[1.0, 3.5], [-0.20, 3.0], [-0.90, 0.90]])                            
        self.simulator_args = ['A1','g2','rho']                                       
        self.K = 3                                                                       # number of parameters
        
        self.stat = 'raw'
        
        self.A1 = 3
        self.B1 = 1
        self.g1 = 2
        self.k1 = 0.5
        self.A2 = 3
        self.B2 = 1
        self.g2 = 0.5
        self.k2 = 0.5
        self.rho = 0.75 
        
        self.invnet = InversionNet()

    def get_true_theta(self):
        return np.array([self.A1, self.g2, self.rho])

     # A. (normalized) marginal quantiles
    def _ss_quantiles(self, X, n_quantiles):
        dim = X.shape[1]
        prob = np.linspace(0.025, 0.975, n_quantiles)
        stat = np.zeros([1, n_quantiles*dim])
        for k in range(dim):
            quantiles = stats.mstats.mquantiles(X[:, k], prob)
            stat_k = quantiles
            stat[0, k*n_quantiles : (k+1)*n_quantiles] = np.array(stat_k)
        return stat

    # B. correlation between the quantiles
    def _ss_corr(self, X):
        U = np.zeros(X.shape)
        n, dim = X.shape
        for d in range(2): 
            noise = 1e-8*np.random.randn(self.n)
            rank = stats.rankdata(X[:, d] + noise).astype(float)
            CDF = rank/n
            U[:, d] = 0.005 + CDF*0.990
        Z = U
        
        V = np.mat(Z).T * np.mat(Z) / Z.shape[0]
        (d,d) = V.shape
        upper_tri_elements = V[np.triu_indices(d, k=1)]
        stat = np.array(upper_tri_elements)
        return stat
    
    def statistics(self, data, theta=None):
        if self.stat == 'raw':
            # (marginal quantiles) + (latent correlation) as summary statistics
            stat_A = self._ss_quantiles(data, n_quantiles=10)
            stat_B = self._ss_corr(data)
            stat = np.hstack((stat_A, stat_B))
            return stat
        else:
            # (marginal quantiles) + (latent correlation) as summary statistics
            stat_A = self._ss_quantiles(data, n_quantiles=5)
            stat_B = self._ss_corr(data)
            stat = np.hstack((stat_A, stat_B))
            return stat
        
    def simulator(self, theta):
        # get the params
        A1,B1,g1,k1 = theta[0], self.B1,  self.g1,   self.k1
        A2,B2,g2,k2 = self.A2,  self.B2,  theta[1],  self.k2
        rho = theta[2]
        V = np.array([[1, rho],[rho, 1]])
        
        # sample z ~ N(0, V)
        Z = distributions.normal_nd.draw_samples([0, 0], V, self.n)
        Z = np.clip(Z, -4, 4)

        # convert z to x
        X = np.zeros(Z.shape)
        Z = torch.tensor(Z).float()
        X[:,0] = self.Z2X(Z[:,0], A1,B1,g1,k1)
        X[:,1] = self.Z2X(Z[:,1], A2,B2,g2,k2)
        return X

    def sample_from_prior(self):
        sample_thetas = []
        for k in range(self.K):
            sample_theta = self.prior[k].draw_samples(self.prior_args[k, 0], self.prior_args[k, 1],  1)[0]
            sample_thetas.append(sample_theta)
        return np.array(sample_thetas)
    
    def tensor(self, X):
        return torch.tensor(X).float()
    
    def Z2X(self, z,A,B,g,k):
        # preparation
        c = 0.80
        g,k = torch.tensor(g).float(), torch.tensor(k).float()
        A,B = torch.tensor(A).float(), torch.tensor(B).float()
        # forward
        w = (1-torch.exp(-g*z))/(1+torch.exp(-g*z))
        v = z*(1+z*z).pow(k)
        x = A + B*(1 + c*w)*v
        return x
    
    def X2Z(self, x, A,B,g,k):
        # preparation
        x = self.tensor(x)
        n = len(x)
        A, B = self.tensor(A).repeat(n,1), self.tensor(B).repeat(n,1)
        g, k = self.tensor(g).repeat(n,1), self.tensor(k).repeat(n,1)
        # [A] inversion-by-neural approximation
        self.invnet.eval()
        xy = torch.cat([x.view(n,-1), A,B,g,k], dim=1)
        z = self.invnet(xy)
        # [B] inversion-by-optimization
        z = z.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=1e-2)
        t0 = time.time()
        for t in range(1000):
            optimizer.zero_grad()
            x = x.view(n, -1)
            x_rec = self.Z2X(z, A,B,g,k).view(n, -1)
            w = torch.exp(-z**2).detach().view(n, -1)    # <-- weight z by p(z)
            loss1 = torch.norm(x_rec - x, dim=1)**2
            loss2 = torch.norm(x_rec - x, dim=1)**2
            loss = (w*loss1).mean() + (w*loss2).max()
            loss.backward()
            optimizer.step()
            if loss1.mean().item()**0.5 < 0.002 and loss2.max().item()**0.5 < 0.05: break
        z = z.view(-1)
        t1 = time.time()
        return z.detach().clone().numpy()
            
    def log_det_jacob(self, z, A,B,g,k):
        # preparation
        z = torch.tensor(z).float()
        n = len(z)
        c = 0.80
        g,k = torch.tensor(g).float(), torch.tensor(k).float()
        A,B = torch.tensor(A).float(), torch.tensor(B).float()
        # compute!
        w = (1-torch.exp(-g*z))/(1+torch.exp(-g*z))
        v = z*(1+z**2).pow(k)
        dw_z = 2*g*torch.exp(-g*z)/(1+torch.exp(-g*z))**2
        dv_z = (1+z**2).pow(k) + (2*k*z**2)*(1+z**2).pow(k-1)
        dx_z = B*(c*dw_z*v + (1+c*w)*dv_z)
        log_abs_det = torch.log(dx_z)
        return log_abs_det.view(-1).numpy()
        
    def log_likelihood(self, theta):
        # get the params
        A1,B1,g1,k1 = theta[0], self.B1,  self.g1,  self.k1
        A2,B2,g2,k2 = self.A2,  self.B2,  theta[1], self.k2
        rho = theta[2]
        V = np.array([[1, rho],[rho, 1]])
        
        # invert to get z
        x = self.data_obs
        Z = np.zeros((self.n, 2))
        Z[:, 0] = self.X2Z(x[:,0], A1,B1,g1,k1)
        Z[:, 1] = self.X2Z(x[:,1], A2,B2,g2,k2)  
        
        # compute Jacobian: det|dx/dz|
        log_det1 = self.log_det_jacob(Z[:, 0], A1,B1,g1,k1)
        log_det2 = self.log_det_jacob(Z[:, 1], A2,B2,g2,k2)
        
        # compute base density: p(z)
        log_pz = stats.multivariate_normal.logpdf(Z, [0, 0], V)

        # log p(x) = log p(z) - log det|dx/dz
        log_px = log_pz - (log_det1 + log_det2)     
        return log_px.sum()
        
    def visualize(self, dim=0):
        print('p(x)=')
        x = np.linspace(-2, 16, 100)
        if dim==0:
            A,B,g,k = self.A1, self.B1, self.g1, self.k1
        else:
            A,B,g,k = self.A2, self.B2, self.g2, self.k2
        z = self.X2Z(x, A,B,g,k)
        log_det = self.log_det_jacob(z,A,B,g,k)
        log_pz = distributions.normal.logpdf(z, 0, 1)
        log_px = log_pz.flatten() - log_det.flatten()
        px = np.exp(log_px)
        plt.plot(x, px)
    
    def train_inversion_net(self):
        invnet = InversionNet()
        invnet.train()
        n = 50000
        z = distributions.uniform.draw_samples(-4.2, 4.2, n)
        z = torch.tensor(z).float()
        xy = np.zeros((n, 5))
        for i in range(n):
            theta = self.sample_from_prior()
            A, B, g, k = theta[0], self.B1, theta[1], self.k1
            x = self.Z2X(z[i], A, B, g, k)
            xy[i] = [x, A, B, g, k]
        xy,z = self.tensor(xy), self.tensor(z)
        invnet.learn(xy, z)
        self.invnet = invnet
        
    def test_inversion_net(self):
        n = 5000
        z = distributions.uniform.draw_samples(-4, 4, n)
        z = torch.tensor(z).float()
        xy = np.zeros((n, 5))
        self.invnet.eval()
        for i in range(n):
            theta = self.sample_from_prior()
            A, B, g, k = theta[0], self.B1, theta[1], self.k1
            x = self.Z2X(z[i], A, B, g, k)
            xy[i] = [x, A, B, g, k]
        xy,z = self.tensor(xy), self.tensor(z)
        z_rec = self.invnet(xy)
        x,A,B,g,k = xy[:,0:1], xy[:,1:2], xy[:,2:3], xy[:,3:4], xy[:,4:5]
        x_rec = self.Z2X(z_rec, A,B,g,k)
        loss_z = torch.norm(z_rec.view(n, -1) - z.view(n, -1), dim=1)
        loss_x = torch.norm(x_rec.view(n, -1) - x.view(n, -1), dim=1)
        
        print('z=', z[0:3].view(-1))
        print('z_rec=', z_rec[0:3].view(-1))
        print('loss(z).mean=', loss_z.mean().item())
        print('loss(z).max=', loss_z.max().item())

        print('x=', x[0:3].view(-1))
        print('x_rec=', x_rec[0:3].view(-1))
        print('loss(x).mean=', loss_x.mean().item())
        print('loss(x).max=', loss_x.max().item())
        
    def fit_approx_likelihood(self, true_samples):
        samples = true_samples
        n, d = samples.shape
        mu = samples.mean(axis=0, keepdims=True)
        M = np.mat(samples - mu)
        cov = 1.0*np.matmul(M.T, M)/n
        mu = torch.Tensor(mu)
        cov = torch.Tensor(cov)
        self.gaussian_approx = torch.distributions.multivariate_normal.MultivariateNormal(mu, cov)
        
    def log_approx_likelihood(self, theta): 
        return self.gaussian_approx.log_prob(torch.Tensor(theta)).item()
    
    def sample_approx_likelihood(self, n): 
        samples = []
        for i in range(n): samples.append(self.gaussian_approx.sample())
        return torch.cat(samples, dim=0).cpu().numpy()
                

        
           
## -------------------------------------------------------------------------------------------- ##

    
    
class InversionNet(torch.nn.Module):
    
    '''
    The network used to invert g-and-k quantile function approximately
    '''

    def __init__(self):
        super().__init__()
        self.bs = 500
        self.lr = 1e-3
        self.wd = 0e-5
        self.positive = False      # <--- setting positiveness does not work; need further checking
        self.main_ABGK = torch.nn.Sequential(PosLinear(4, 100, False),
                                   Tanh2(100, False),
                                   PosLinear(100, 100, False),
                                   )
        self.main_x = torch.nn.Sequential(
                                   PosLinear(1, 100, self.positive),
                                   Tanh2(100, self.positive),
                                   PosLinear(100, 100, self.positive),
                                   )
        self.merge = torch.nn.Sequential(PosLinear(200, 100, self.positive),
                                   Tanh2(100, self.positive),
                                   PosLinear(100, 1, self.positive),
                                   )
                            
    def forward(self, xy):
        n, d = xy.size()
        x = xy[:, 0:1]
        x = self.main_x(x)
        ABGK = xy[:, 1:]
        ABGK = self.main_ABGK(ABGK)
        xABGK = torch.cat([x, ABGK], dim=1)
        out = self.merge(xABGK)
        return out
            
    def loss_func(self, xy, z):
        n, d = xy.size()
        z = z.view(n, 1)
        z_pred = self.forward(xy).view(n, 1)
        J = torch.norm(z - z_pred, dim=1)**2
        return J.mean()
        
    def learn(self, x, y):    
        # hyperparams 
        T = 3000
        T_NO_IMPROVE_THRESHOLD = 500
        
        # divide train & val 
        n = len(x)
        n_train = int(0.85*n)
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
                loss = self.loss_func(x_chunks[i], y_chunks[i])
                loss.backward()
                optimizer.step()
            end = time.time()

            # early stopping if val loss does not improve after some epochs
            self.eval()
            loss_val = torch.zeros(1, device=x.device)
            for j in range(len(x_v_chunks)):
                loss_val += self.loss_func(x_v_chunks[j], y_v_chunks[j])/len(x_v_chunks)
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
    
    
class PosLinear(torch.nn.Module):
    
    '''
    A Linear module whose weights are all positive
    '''

    def __init__(self, in_features, out_features, positive=True):
        super().__init__()
        self.positive = positive
        self.linear = torch.nn.Linear(in_features, out_features, bias=True)
        self.softplus = torch.nn.Softplus(beta=10)
        
    def forward(self, inputs):
        if self.positive:
            weight = self.softplus(self.linear.weight) 
            out = F.linear(inputs, weight, self.linear.bias)
        else:
            weight = self.linear.weight
            out = F.linear(inputs, weight, self.linear.bias)
        return out
    
    
class Tanh2(torch.nn.Module):
    
    '''
    Modified Tanh such that it is more flexible
    '''

    def __init__(self, in_features, positive=False):
        super().__init__()
        self.positive = positive
        self.a = torch.nn.Parameter(torch.randn((1, in_features)))
        
    def forward(self, x):
        if self.positive:
            return x + F.tanh(self.a)*F.tanh(x)
        else:
            return F.tanh(x)
        
        
        
## -------------------------------------------------------------------------------------------- ##


# class Gaussian(torch.nn.Module):
#     def __init__(self, dim, scaling_factor=1.0):
#         super(Gaussian, self).__init__()
#         self.mu = torch.nn.Parameter(torch.Tensor(1, dim))
#         self.V = torch.nn.Parameter(torch.Tensor(dim, dim))
#         self.scaling_factor = scaling_factor
        
#     def fit(self, samples):
#         [n, dim] = samples.shape
#         mu = samples.mean(axis=0, keepdims=True)
#         M = np.mat(samples - mu)
#         cov = self.scaling_factor*np.matmul(M.T, M)/n
#         self.mu.data = torch.Tensor(mu)
#         self.V.data = torch.Tensor(cov)
#         self.mvn = torch.distributions.multivariate_normal.MultivariateNormal(self.mu, self.V)

#     def log_probs(self, x):
#         return self.mvn.log_prob(x).view(-1)
    
#     def sample(self):
#         return self.mvn.sample()
        