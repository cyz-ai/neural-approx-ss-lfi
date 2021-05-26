from abc import ABCMeta, abstractmethod

import numpy as np
import os, sys, time, math
import scipy.stats as stats
import matplotlib.pyplot as plt

import utils_math, utils_os
import distributions
import discrepancy
import algorithms.ABC_algorithms as ABC_algorithms


class SMC_ABC(ABC_algorithms.Base_ABC):

    '''
    Sequantial Monte Carlo ABC
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
        super(SMC_ABC, self).__init__(problem, discrepancy, hyperparams, **kwargs)

        self.needed_hyperparams = ['epsilon']
        self.epsilon = hyperparams.epsilon
        self.posterior_array = []
        
        # compute pi(theta)
        self.volume = 1.0
        ranges = self.problem.prior_args
        for k in range(self.problem.K): self.volume = self.volume*(ranges[k,1] - ranges[k,0])
          
    def sort_samples(self):
        
        # > Sort the samples according to the corresponding |s - s^o|
        
        y_obs = self.convert_stat(self.y_obs)
        all_stats = self.convert_stat(np.vstack(self.all_stats))
        all_samples = np.vstack(self.all_samples)
        [m, d] = all_stats.shape
        [m, K] = all_samples.shape
        all_discrepancies = np.zeros(m)
        for i in range(m): 
            all_discrepancies[i] = discrepancy.eculidean_dist(y_obs, all_stats[i])
        self.rej_stats = np.zeros((self.num_samples, d), float)
        self.rej_samples = np.zeros((self.num_samples, K), float)
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
            cov = np.cov(M.T)
            return [mu, cov]
        else:
            # Copula
            copula = distributions.copula()
            copula.fit(samples)
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
        log_weight_array = np.zeros((40000))
        for i in range(40000):
            theta = self._sample(self.fake_posterior)
            pdf_fake_prior = self.pdf_fake_prior(theta)
            log_weight_array[i] = -np.log(pdf_fake_prior)
            if np.isnan(pdf_fake_prior):
                print('nan!', pdf_fake_prior)
        log_max_weight = np.max(log_weight_array)  
        thetas = []
        while len(thetas)<=2000:
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
        
    def pdf_fake_prior(self, theta):
        '''
           > p(theta) = 1/n * ∑ q_r(theta|x^o)
        '''
        pdf = 1.0/self.volume
        posterior_array = self.posterior_array[0:self.l]
        L = len(posterior_array) + 1.0
        for posterior in posterior_array: pdf += self._pdf(theta, posterior)
        return pdf/L
                
    def log_likelihood(self, theta):
        '''
           > log_q_r(theta|x^o)
        '''
        return np.log(self._pdf(theta, self.posterior))
      
    def sample_from_true_posterior(self):
        '''
           > theta ~ q_r(theta|x^o)
        '''
        return self._sample(self.posterior)
          
    def run(self):
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
        self.all_discrepancies = []
        for l in range(L):
            print('iteration ', l)
            self.l = l
            self.simulate()
            self.all_stats.append(self.stats)
            self.all_samples.append(self.samples)
            self.all_discrepancies.append(self.discrepancies)
            self.sort_samples()
            self.learn_fake_posterior()
            self.learn_true_posterior()
            self.prior = self.sample_from_true_posterior   
            print('\n')
        self.save_results()