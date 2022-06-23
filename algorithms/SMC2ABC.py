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


class SMC2_ABC(ABC_algorithms.Base_ABC):

    '''
    Sequantial Monte Carlo ABC (with learned sufficient stat).
    '''

    def __init__(self, problem, discrepancy, hyperparams, **kwargs):
        '''
            problem : ABC_Problem instance
                The problem to solve An instance of the ABC_Problem class.
            discrepency: function pointer
                The data discrepency
            hyperparams: object
                The hyper-parameters (accessed by .member_name)
        '''
        super(SMC2_ABC, self).__init__(problem, discrepancy, hyperparams, **kwargs)

        self.needed_hyperparams = ['epsilon']
        self.epsilon = hyperparams.epsilon
        self.device = torch.device(hyperparams.device)
        self.stat_net = None 
        self.stat_array = []
        self.posterior_array = []

        # compute p(theta)
        self.volume = 1.0
        ranges = self.problem.prior_args
        for k in range(self.problem.K): self.volume = self.volume*(ranges[k,1] - ranges[k,0])
            
    def convert_stat(self, x): 
        # no autoencoder, directly return s
        if self.stat_net is None: 
            s = x
            return s
        # convert raw data to summary stat: s = S(x)
        else:
            s = self.stat_net.encode(torch.tensor(x).float())
            return s.detach().cpu().numpy()
                
    def learn_stat(self):
        print('\n > fitting encoder')
        all_stats = torch.tensor(np.vstack(self.all_stats)).float().to(self.device)
        all_samples = torch.tensor(np.vstack(self.all_samples)).float().to(self.device)
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
          
    def sort_samples(self):
        
        # > Sort the samples according to the corresponding |s - s^o|
        
        all_stats = np.vstack(self.all_stats)
        all_samples = np.vstack(self.all_samples)
        all_discrepancies = self.recompute_discrepancy(all_stats)
        self.rej_stats = np.zeros((self.num_samples, self.y_obs.shape[1]), float)
        self.rej_samples = np.zeros((self.num_samples, self.num_theta), float)
        idxes = np.argsort(all_discrepancies)
        for i in range(self.num_samples):
            idx = idxes[i]
            self.rej_stats[i, :], self.rej_samples[i, :] = all_stats[idx, :], all_samples[idx, :]
                    
    def _fit(self, samples):
        if self.l < self.hyperparams.L-1:
            # Gaussian
            [n, dim] = samples.shape
            mu = np.mean(samples, axis=0)
            M = np.mat(samples - mu)
            cov = np.matmul(M.T, M)/n
            return [mu, cov]
        else:
            # Copula
            while True:
                copula = distributions.copula()
                copula.fit(samples)
                cov_diagonal = copula.gc_cov.diagonal()
                print('cov_diagonal=', cov_diagonal)
                if np.abs(cov_diagonal.mean() - 1.0) < 0.20: break
            return [copula]
        
    def _sample(self, distribution):
        if len(distribution) == 2:
            # Gaussian
            mu, cov = distribution[0], distribution[1]
            theta = distributions.normal_nd.draw_samples(mu, cov, 1)
        else:
            # Copula
            copula = distribution[0]
            theta = copula.sample()
        return theta.flatten()
        
    def _pdf(self, sample, distribution):
        if len(distribution) == 2:
            mu, cov = distribution[0], distribution[1]
            log_pdf = distributions.normal_nd.logpdf(sample, mu, cov)
        else:
            copula = distribution[0]
            log_pdf = copula.logpdf(sample)
        return np.exp(log_pdf)
     
    def learn_fake_posterior(self):
        '''
            p_r(theta|x^o) ∝ p(theta)p(x^o|theta)
        '''
        print('> learning fake posterior ')
        self.fake_posterior = self._fit(self.rej_samples)
        
    def learn_true_posterior(self):
        '''
            > q_r(theta|x^o) ∝ pi(theta)/p(theta) * p_r(theta|x^o)
        '''
        # [A] sample theta ~ q_r(theta|x^o) 
        print('> learning true posterior ')
        log_weight_array = np.zeros((10000))
        for i in range(10000):
            theta = self._sample(self.fake_posterior)
            pdf_fake_prior = self.pdf_fake_prior(theta)
            log_weight_array[i] = -np.log(pdf_fake_prior)
        log_max_weight = np.max(log_weight_array)  
        thetas = []
        while len(thetas)<=500:
            theta = self._sample(self.fake_posterior)
            pdf_fake_prior = self.pdf_fake_prior(theta)
            log_weight = -np.log(pdf_fake_prior)
            prob_accept = log_weight - log_max_weight
            u = distributions.uniform.draw_samples(0, 1, 1)[0]
            if np.log(u) < prob_accept: thetas.append(theta)
        thetas = np.vstack(thetas)   
        # [B]. fit p_{r+1}(theta) with the sampled thetas
        self.posterior = self._fit(thetas)
        self.posterior_array.append(self.posterior)
                        
    def log_likelihood(self, theta):
        '''
           > log_q_r(theta|x^o)
        '''
        return np.log(self._pdf(theta, self.posterior))
    
    def pdf_fake_prior(self, theta):
        '''
           > p(theta) = 1/n * ∑ q_r(theta|x^o)
        '''
        pdf = 1.0/self.volume
        posterior_array = self.posterior_array[0:self.l]
        L = len(posterior_array) + 1.0
        for posterior in posterior_array: pdf += self._pdf(theta, posterior)
        return pdf/L
    
    def log_fake_posterior(self, theta):
        '''
           > log_p_r(theta|x^o)
        '''
        return np.log(self._pdf(theta, self.fake_posterior))
    
    def sample_from_true_posterior(self):
        '''
           > theta ~ q_r(theta|x^o)
        '''
        return self._sample(self.posterior)
                
    def recompute_discrepancy(self, stats):
        new_stats = torch.tensor(self.convert_stat(stats)).float()
        new_stats = new_stats.detach().cpu().numpy()
        y_obs = torch.tensor(self.convert_stat(self.y_obs)).float()
        y_obs = y_obs.detach().cpu().numpy()
        [n, dim] = new_stats.shape
        discrepancies = np.zeros(n)
        for i in range(n):
            y = new_stats[i]
            discrepancies[i] = self.discrepancy(y_obs, y)
        return discrepancies
    
    def run(self, all_stats=None, all_samples=None):
        '''
           > main pipeline for the algorithm
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
            if all_stats is None:
                self.simulate()
                self.all_stats.append(self.stats)
                self.all_samples.append(self.samples)
            else:
                self.all_stats = all_stats
                self.all_samples = all_samples
            self.learn_stat()
            self.sort_samples()
            self.learn_fake_posterior()
            self.learn_true_posterior()
            self.prior = self.sample_from_true_posterior   
            print('\n')
        self.save_results()