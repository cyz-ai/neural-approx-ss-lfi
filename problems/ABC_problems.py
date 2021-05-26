import numpy as np
import math
import distributions
import utils_os, utils_math
from abc import ABCMeta, abstractmethod

class ABC_Problem(object):

    __metaclass__ = ABCMeta

    # The dimensionality of y_obs
    y_dim = None
    # The value of y_obs
    y_obs = None
    data_obs = None


    # Prior distribution class
    prior = None
    # Prior distribution arguments
    prior_args = None

    # The range for the problem (used for visualisation)
    rng = None

    # The true posterior distribution class
    true_posterior = None
    # The range of the true posterior distribution
    true_posterior_rng = None
    # The arguments of the true posterior distribution
    true_posterior_args = None

    # List of labels of simulator arguments (what each dimension of theta is)
    simulator_args = None

    # The true arguments used to obtain the y_obs
    true_args = None

    @abstractmethod
    def get_true_theta(self):
        '''
        Returns the true parameter setting.
        '''
        return NotImplemented

    @abstractmethod
    def statistics(self, data, theta=None):
        '''
        Compute the summary statistics of the simulated data.
        ----------
        Parameters
        ----------
        data : array
            The array of values returned by the simulator.
        Returns
        -------
        stats : array
            The array of computed statistics.
        '''
        return NotImplemented

    @abstractmethod
    def simulator(self, theta):
        '''
        Runs the simulator with the given parameter settings.
        ----------
        Parameters
        ----------
        theta : array
            The array of parameters
        Returns
        -------
        output : array
            The simulator output
        '''
        return NotImplemented

    @abstractmethod
    def log_likelihood(self, theta):
        '''
        Calculate the log_likelihood of theta given y_obs.
        ----------
        Parameters
        ----------
        theta : array
            The array of parameters
        Returns
        -------
        output : float
            The likelihood value L(theta) in [0,1]
        '''
        return NotImplemented

    @abstractmethod
    def sample_from_prior(self):
        '''
        Sample one value from the prior.
        ----------
        Returns
        -------
        output : array
            one sample from the prior
        '''
        return NotImplemented

    def sample_from_true_posterior(self):
        '''
        Sample the parameters from the true posterior (by likelihood-based rejection sampling).
        Returns
        -------
        output : array
            N samples from the posterior
        '''

        # some preparison
        num_theta = self.K
        true_posterior_samples = np.zeros((self.N, num_theta), float)
        self.current_sim_calls = 0

        # find max(L(theta))
        print('[True posterior] pilot run ')
        max_ll = -math.inf
        for j in range(1, 50*self.N):
            theta = self.sample_from_prior()
            ll = self.log_likelihood(theta)
            if ll > max_ll: max_ll = ll
            if j%(self.N)==0: print('finished pilot run:', j)

        # rejection sampling
        i = 0
        while i < self.N:
            self.current_sim_calls = self.current_sim_calls + 1

            theta = self.sample_from_prior()
            prob_accept = self.log_likelihood(theta) - max_ll
            u = distributions.uniform.draw_samples(0, 1, 1)[0]

            if np.log(u) > prob_accept:
                continue

            true_posterior_samples[i, 0:] = theta
            i = i + 1

            if i % int(self.N/10) == 0 and i>1:
                print('[True posterior] finished sampling ', i)

        return true_posterior_samples

    def theta2prob(self, theta):

        # > convert theta -> normalized theta (assuming uniform prior)

        prob = np.zeros(theta.shape)
        (n, dim) = theta.shape
        for k in range(dim):
            prob[:,k] = (theta[:,k] - self.prior_args[k,0])/(self.prior_args[k,1] - self.prior_args[k,0])
        return prob

    def prob2theta(self, prob):

        # > convert normalized theta -> theta (assuming uniform prior)

        theta = np.zeros(prob.shape)
        (n, dim) = prob.shape
        for k in range(dim):
            theta[:,k] = self.prior_args[k,0] + (self.prior_args[k,1] - self.prior_args[k,0])*prob[:,k]
        return theta