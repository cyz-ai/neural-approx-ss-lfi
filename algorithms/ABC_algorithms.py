from abc import ABCMeta, abstractmethod

import numpy as np
import os, sys, time, math
import scipy.stats as stats
import matplotlib.pyplot as plt

import utils_math, utils_os
import distributions
import discrepancy



class Hyperparams(object):

    '''
    Hyper-parameters for ABC algorithms.
    '''

    __metaclass__ = ABCMeta

    def __init__(self):
        self.epsilon = None
        self.num_samples = 1000
        self.pilot_run_N = 100000
        self.whiten = False
        self.save = False
        self.num_sim_division = None


class Base_ABC(object):

    '''
    Abstract base class for ABC algorithms.
    '''

    __metaclass__ = ABCMeta

    def __init__(self, problem, discrepancy, hyperparams, verbose=False, save=True, **kwargs):

        self.problem = problem
        
        self.y_obs = problem.statistics(problem.data_obs, problem.get_true_theta())
        self.y_dim = self.y_obs.size

        self.simulator = problem.simulator
        self.prior = problem.sample_from_prior
        self.prior_args = problem.prior_args
        self.num_theta = self.problem.K
        
        self.discrepancy = discrepancy
        self.hyperparams = hyperparams
        
        self.save_dir = hyperparams.save_dir

        self.samples = None
        self.stats = None
        
        self.prior = problem.sample_from_prior
        self.num_sim = hyperparams.num_sim
        self.num_samples = hyperparams.num_samples

    def __repr__(self):
        s = type(self.problem).__name__ + '_' + type(self).__name__
        return s

    def get_parameters(self):
        '''
        Returns the list of parameter values. Order is determined by
        `self.needed_params`.
        '''
        return [self.__dict__[par] for par in self.needed_params]

    def save_results(self):
        '''
        Saves the result_data of this algorithm.
        Note: Should be called after a call to `run()`
        '''
        utils_os.save_algorithm(self.save_dir, self)

    def whiten(self, x):

        # > whiten the summary statistics

        if self.hyperparams.whiten is True:
            return (x-self.mean)/np.diag(self.COV)**0.5
        else:
            return x

    def convert_stat(self, x):

        # > concert the summary stats if necessary

        return x

    def is_valid_theta(self, theta):

        # > check if the theta is within the prior range

        flag = True
        for k in range(len(theta)):
            if theta[k] < self.problem.prior_args[k, 0] or theta[k] > self.problem.prior_args[k, 1]:
                flag = False
        return flag
    
    def determine_whiten(self):

        # > pilot run function that determines the scaled of each statistics

        print('[ABC] pilot run: whitening')

        # Record current settings
        whiten = self.hyperparams.whiten 
        num_sim = self.num_sim
        
        # Do simulation
        self.hyperparams.whiten  = False
        self.num_sim = self.hyperparams.pilot_run_N
        self.simulate()
        
        # Compute mean & covariance matrix
        stats = self.stats
        stats = np.mat(stats)
        mean = stats.mean(axis=0)
        cov = (stats-mean).T * (stats-mean) / stats.shape[0]
        self.COV = cov
        self.mean = mean

        # Save results
        utils_os.save_object(self.save_dir, 'pilot_run_whiten_cov.npy', self.COV)
        utils_os.save_object(self.save_dir, 'pilot_run_whiten_mean.npy', self.mean)
        
        # Recover settings
        self.hyperparams.whiten = whiten
        self.num_sim = num_sim
        

    def determine_epsilon(self):

        # > pilot run function that determines the quantiles of epsilon

        print('[ABC] pilot run: epsilon')

        # Record current settings
        num_sim = self.num_sim
        
        # Do simulation
        self.num_sim = self.hyperparams.pilot_run_N
        self.simulate()

        # Sort the discrepancy list
        discrepancies = self.discrepancies
        discrepancies.sort()
        self.discrepancies = discrepancies

        # Save results
        utils_os.save_object(self.save_dir, 'pilot_run_epsilon.npy', discrepancies)
        
        # Recover settings
        self.num_sim = num_sim

    def pilot_run(self):

        # > pilot run that determines (a) the scale of summary stat (b) the range of epsilon
        
        if utils_os.is_file_exist(self.save_dir, 'pilot_run_epsilon.npy'):
            print('[ABC] already completed pilot run')
            self.mean = utils_os.load_object(self.save_dir, 'pilot_run_whiten_mean.npy')
            self.COV = utils_os.load_object(self.save_dir, 'pilot_run_whiten_cov.npy')
            self.discrepancies = utils_os.load_object(self.save_dir, 'pilot_run_epsilon.npy')
            return
        else:
            print('[ABC] running pilot run')
            self.determine_whiten()
            self.determine_epsilon()
            
    def simulate(self):

        # > wrapper function of rejection sampling algorithm. Launch several processes to do sampling in parallel

        # Initialization
        self.stats = np.zeros((self.num_sim, self.y_dim), float)
        self.samples = np.zeros((self.num_sim, self.num_theta), float)
        self.discrepancies = []

        # Run in parallel
        params = []
        N_proc = 4                                     
        num_budget_per_proc = int(self.num_sim/N_proc)
        num_samples_per_proc = int(self.num_sim/N_proc)                                                                                                                                                                                
        for k in range(N_proc): params.append([num_budget_per_proc, num_samples_per_proc, k])
        func = self._simulate
        rets = utils_os.run_in_parallel(func, params, N_proc)
        
        # Concatenate the results
        for k in range(N_proc):
            ret = rets[k]
            samples, stats, discrepancies = ret[0], ret[1], ret[2]
            [n, dim] = samples.shape
            self.samples[k*n:(k+1)*n, :] = samples
            self.stats[k*n:(k+1)*n, :] = stats
            self.discrepancies += discrepancies
            
    def _simulate(self, param):

        # > true implementation of rejection sampling

        print('[ABC] sub-process start!')

        # Extract params
        num_sim, num_samples, pid = param[0], param[1], param[2]

        stats, samples = np.zeros((num_sim, self.y_dim), float), np.zeros(
                (num_sim, self.num_theta), float)
        discrepancies = []

        # Simulate
        np.random.seed(pid + int(time.time()))
        for i in range(num_sim):
            while True:
                # Sample from the prior distribution
                theta = self.prior()

                # Throw away bad thetas
                if self.is_valid_theta(theta) is False: continue

                # Get summary statistics
                data = self.problem.simulator(theta)
                if data is None: continue
                y = self.problem.statistics(data=data, theta=theta)

                # Whitening stat
                y, y_obs = self.whiten(y), self.whiten(self.y_obs)

                # Calculate error
                error = self.discrepancy(y_obs, y)

                # Collect samples & discrepancies
                stats[i, :] = y
                samples[i, :] = theta
                discrepancies.append(error)
                break

            if i % (int(num_sim/5)) == 0 and i>=1 and pid==0:
                print('[sampling] finished sampling ', i)
                
        return [samples, stats, discrepancies]

    @abstractmethod
    def run(self, num_samples, reset=True):
        return NotImplemented