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
import algorithms.SMCABC as SMCABC



class TP_ABC(SMCABC.SMC_ABC):

    '''
    True posterior approximation via <rejection ABC + known sufficient stat>, used in Ising model
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
        super(TP_ABC, self).__init__(problem, discrepancy, hyperparams, **kwargs)
        
    def convert_stat(self, x):
        x = np.atleast_2d(x)
        n = len(x)
        stats = []
        for i in range(n):
            stat = self.problem.sufficient_stat(x[i])
            stats.append(stat)
        s = np.vstack(stats)
        return s
    
    def get_true_samples(self):
        return self.rej_samples
    
    def run(self, rej_samples=None):
        '''
           > main pipeline for the algorithm
        '''
        
        # initialization
        self.prior = self.problem.sample_from_prior
        total_num_sim = self.num_sim
        
        # one round learning
        ratios = [1.0, 0.0]
        self.l = 0
        self.all_stats = []
        self.all_samples = []
        self.all_discrepancies = []
        self.num_sim = int(total_num_sim*ratios[self.l]) 
        if rej_samples is None:
            self.simulate()
            self.all_stats.append(self.stats)
            self.all_samples.append(self.samples)
            self.all_discrepancies.append(self.discrepancies)
            self.sort_samples()
        else:
            self.rej_samples = rej_samples
        self.learn_fake_posterior()
        self.learn_true_posterior()
        print('\n')
        self.save_results()
        
        
        
