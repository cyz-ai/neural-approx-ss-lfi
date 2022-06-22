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
from nde import MAF,MDN
from copy import deepcopy


class SNL_ABC(ABC_algorithms.Base_ABC):

    '''
    Sequential neural likelihood.
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
        super(SNL_ABC, self).__init__(problem, discrepancy, hyperparams, **kwargs)
        
        self.needed_hyperparams = ['epsilon']
        self.epsilon = hyperparams.epsilon
        self.device = torch.device(hyperparams.device)
        self.nde_net = None
        self.nde_array = []

    def fit_nde(self):
        all_stats = torch.tensor(np.vstack(self.all_stats)).float().to(self.device)
        all_samples = torch.tensor(np.vstack(self.all_samples)).float().to(self.device)
        [n, dim] = all_stats.size()
        if not hasattr(self.hyperparams, 'nde') or self.hyperparams.nde == 'MAF':
            net = MAF.MAF(n_blocks=5, n_inputs=dim, n_hidden=50, n_cond_inputs=self.problem.K)
        else:
            net = MDN.MDN(n_in=self.y_obs.shape[1], n_hidden=50, n_out=self.problem.K, K=8)
        net.train().to(self.device)
        net.learn(inputs=all_stats, cond_inputs=all_samples)
        net = net.cpu()
        self.nde_net = net
        self.nde_array.append(net)
        
    def sample_from_nde(self):
        '''
            theta ~ q(theta|x_o) 
        '''
        net = self.nde_net
        net.eval()
        # pilot run to determine max log-likelihood
        if self.max_ll is None:
            self.max_ll = -math.inf
            for j in range(40000):
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
    
    def log_likelihood(self, theta):
        '''
            log p(theta|x_o) = log q(x_o|theta)     (note: uniform prior)
        '''
        net = self.nde_net
        net.eval()
        y_obs, theta = self.convert_stat(self.whiten(self.y_obs)), theta
        y_obs, theta = torch.tensor(y_obs).float(), torch.tensor(theta).float().view(1, -1)
        log_probs = net.log_probs(inputs=y_obs, cond_inputs=theta)
        return log_probs.item()
    
    def set(self, l=0):
        self.l = l
        self.nde_net = self.nde_array[l]
                
    def run(self):

        # > main pipeline for the algorithm

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
            self.simulate()
            self.all_stats.append(self.stats)
            self.all_samples.append(self.samples)
            self.fit_nde()
            self.prior = self.sample_from_nde   
            print('\n')
        self.num_sim = total_num_sim
        
        # return
        self.save_results()

        
        
        
        
        
