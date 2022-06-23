from abc import ABCMeta, abstractmethod

import numpy as np
import torch 
import os, sys, time, math
import scipy.stats as stats
import matplotlib.pyplot as plt

import utils_math, utils_os
import distributions
import discrepancy
import algorithms.ABC_algorithms as ABC_algorithms
from nn import ISN, MSN, SSN
from nde import MAF, MDN
from copy import deepcopy


class SNL2_ABC(ABC_algorithms.Base_ABC):

    '''
    Sequential neural likelihood (with summary stat).
    '''
    
    def __init__(self, problem, discrepancy, hyperparams, **kwargs):
        '''
        Creates an instance of rejection ABC for the given problem.
        Parameters
        ----------
            problem : ABC_Problem instance
                The problem to solve An instance of the ABC_Problem class.
            discrepency: function pointer
                The data discrepency
            hyperparams: 1-D array
                The hyper-parameters [epsilon, num-samples]
            verbose : bool
                If set to true iteration number as well as number of
                simulation calls will be printed.
            save : bool
                If True will save the result to a (possibly exisisting)
                database
        '''
        super(SNL2_ABC, self).__init__(problem, discrepancy, hyperparams, **kwargs)
        
        self.needed_hyperparams = ['epsilon']
        self.epsilon = hyperparams.epsilon
        self.device = torch.device(hyperparams.device)
        self.nde_net = None                             # the learned q(x|theta)
        self.stat_net = None                            # the learned s(x)
        self.nde_array = []                             
        self.stat_array = []     
        self.proposal_array = []                        # the proposal used at each round
        self.hyperparams = hyperparams
 
    def convert_stat(self, x): 
        # no autoencoder, directly return s
        if self.stat_net is None: 
            s = x
            return s
        # convert raw data to summary stat: s = S(x)
        else:
            s = self.stat_net.encode(torch.tensor(x).float())
            return s.detach().cpu().numpy()
            
    def fit_nde(self):
        print('\n > fitting nde')
        all_stats = torch.tensor(self.convert_stat(np.vstack(self.all_stats[0:self.l+1]))).float().to(self.device)
        all_samples = torch.tensor(np.vstack(self.all_samples[0:self.l+1])).float().to(self.device)
        [n, dim] = all_stats.size()
        print('all_stats.size()', all_stats.size())
        if self.hyperparams.nde == 'MAF':
            net = MAF.MAF(n_blocks=5, n_inputs=dim, n_hidden=50, n_cond_inputs=self.problem.K)
        if self.hyperparams.nde == 'MDN':
            net = MDN.MDN(n_in=self.problem.K, n_hidden=50, n_out=dim, K=8)
        net.train().to(self.device)
        net.learn(inputs=all_stats, cond_inputs=all_samples)
        net = net.eval().cpu()
        self.nde_net = net
        self.nde_array.append(net)

    def learn_stat(self):
        print('\n > fitting encoder')
        all_stats = torch.tensor(np.vstack(self.all_stats[0:self.l+1])).float().to(self.device)
        all_samples = torch.tensor(np.vstack(self.all_samples[0:self.l+1])).float().to(self.device)
        [n, dim] = all_stats.size()
        h = self.problem.K*2
        print('summary statistic dim =', h, 'original dim =', dim)
        architecture = [dim] + [100, 100, h]    
        print('architecture', architecture)
        if self.hyperparams.stat == 'infomax':
            net = ISN.ISN(architecture, dim_y=self.problem.K, hyperparams=self.hyperparams)
        if self.hyperparams.stat == 'moment':
            net = MSN.MSN(architecture, dim_y=self.problem.K, hyperparams=self.hyperparams)
        if self.hyperparams.stat == 'score':
            net = SSN.SSN(architecture, dim_y=self.problem.K, hyperparams=self.hyperparams)
        net.train().to(self.device)
        net.learn(x=all_stats, y=all_samples)
        net = net.eval().cpu()
        self.stat_net = net
        self.stat_array.append(net)

    def sample_from_nde(self):
        net = self.nde_net
        net.eval()
        # pilot run for rej sampling
        if self.max_ll is None:
            self.max_ll = -math.inf
            for j in range(10000):
                theta = self.problem.sample_from_prior()
                ll = self.log_likelihood(theta)
                if ll > self.max_ll: self.max_ll = ll
        # rejection sampling
        while True:
            theta = self.problem.sample_from_prior()
            prob_accept = self.log_likelihood(theta) - self.max_ll
            u = distributions.uniform.draw_samples(0, 1, 1)[0]
            if np.log(u) < prob_accept: break
        return theta
        
    def log_likelihood(self, theta, use_ratio=False):
        if not use_ratio:
            '''
                log p(theta|x_o) = log q(x_o|theta)     (note: uniform prior)
            '''
            net = self.nde_net
            net.eval()
            y_obs, theta = self.convert_stat(self.whiten(self.y_obs)), theta
            y_obs, theta = torch.tensor(y_obs).float(), torch.tensor(theta).float().view(1, -1)
            log_probs = net.log_probs(inputs=y_obs, cond_inputs=theta)
            return log_probs.item()
        else:
            '''
            log p(theta|x_o) = log r(x_o, theta) + C(x_o)   (note: uniform prior. Here r(x, theta) = p(x, theta)/p(x)p(theta))
            '''
            net = self.stat_net
            net.eval()
            y_obs, theta = self.y_obs, theta
            y_obs, theta = torch.tensor(y_obs).float(), torch.tensor(theta).float().view(1, -1)
            log_probs = net.log_likelihood(y_obs, theta)
            return log_probs.view(-1).item()
        
    def set(self, l=0):
        self.l = l
        self.stat_net = self.stat_array[l]
        self.nde_net = self.nde_array[l]

    def run(self, all_stats=None, all_samples=None):
        '''
            main pipeline for the algorithm
        '''
        # initialization
        self.prior = self.problem.sample_from_prior
        
        # iterations
        L = self.hyperparams.L
        total_num_sim = self.num_sim 
        self.num_sim = int(total_num_sim/L)
        self.all_stats = []
        self.all_samples = []
        for l in range(L):
            print('iteration ', l)
            self.l = l
            self.max_ll = None
            if all_stats is None:
                self.simulate()
                self.all_stats.append(self.stats)
                self.all_samples.append(self.samples)
            else:
                self.all_stats = all_stats
                self.all_samples = all_samples
            self.learn_stat()
            self.fit_nde()
            self.prior = self.sample_from_nde
            print('\n')
        self.num_sim = total_num_sim
        
        # return
        self.save_results()

        
        
        
        
        
