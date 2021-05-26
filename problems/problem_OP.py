import numpy as np
import math
import scipy.stats as stats
from abc import ABCMeta, abstractmethod
import distributions 
import utils_math
from problems import ABC_problems
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time


class OP_Problem(ABC_problems.ABC_Problem):

    '''
    (discretized) Ornstein-Uhlenbeck process with two parameters: dx_t = theta1*(e^(theta2) - x_t) + G*dW_t 
    '''

    def __init__(self, N=100, n=50):

        self.N = N                                                                       # number of posterior samples
        self.n = n                                                                       # length of the data vector x = {x_1, ..., x_n}

        self.prior = [distributions.uniform, distributions.uniform]                      # uniform prior
        self.prior_args = np.array([[0.0, 1.0], [-2.0, 2.0]])                            
        self.simulator_args = ['theta1', 'theta2']                                       
        self.K = 2                                                                       # number of parameters
        self.stat = 'raw'

        self.true_theta1 = 0.5
        self.true_theta2 = 1.0
        self.G = 0.5
        self.T = 10.0                                                                    # total time of the process

    def get_true_theta(self):
        return np.array([self.true_theta1, self.true_theta2])

    def statistics(self, data, theta=None):
        if self.stat == 'raw':
            idx = np.arange(self.n)
            stat = data[idx]
            return np.reshape(stat, (1, len(idx)))
        if self.stat == 'expert':
            # E[x] and V[x]
            mean = np.mean(data)
            std = np.std(data)
        
            # standardize
            x = (data - mean) / std

            # auto correlation coefficient with lag 1,2,3
            ac = []
            for lag in [1, 2, 3]:
                m = self.n-lag
                v = np.dot(x[:-lag].reshape(1, m), x[lag:]).squeeze()/m
                ac.append(v)

            stat = np.array([mean, std] + ac)
            return np.reshape(stat, (1, len(stat)))
    
    def simulator(self, theta):
        # get the params
        theta1 = theta[0]
        theta2 = math.exp(theta[1])

        # noises
        T, d = self.T, self.n+1
        w = np.atleast_2d(distributions.normal.draw_samples(0, 1, self.n)).T
        dt = T/d

        # Euler-Murayama discretization
        x = np.zeros([self.n, 1])
        x[0, :] = 10
        for t in range(self.n-1):
            mu, sigma = theta1*(theta2 - x[t, :])*dt, self.G*(dt**0.5)*w[t]
            x[t+1, :] = x[t,:] + mu + sigma
        return x

    def sample_from_prior(self):
        sample_theta1 = self.prior[0].draw_samples(self.prior_args[0, 0], self.prior_args[0, 1],  1)[0]
        sample_theta2 = self.prior[1].draw_samples(self.prior_args[1, 0], self.prior_args[1, 1],  1)[0]
        return np.array([sample_theta1, sample_theta2])
    
    def log_likelihood(self, theta):
        # get the params
        theta1 = theta[0]
        theta2 = math.exp(theta[1])
        
        # time intervals
        T, d = self.T, self.n+1
        dt = T/d

        # log p(x|phi) = \sum_d log p(x_d|x<d)
        x = self.data_obs[:, 0]
        ll = 0
        for t in range(self.n-1):
            mu, sigma = theta1*(theta2 - x[t])*dt, self.G*(dt**0.5)
            ll_t = distributions.normal.logpdf(x[t+1] - x[t], mu, sigma)
            ll += ll_t
        return ll
        
    def visualize(self):
        plt.figure(figsize=(5,4))
        t = np.linspace(0, self.n, self.n).astype(int)
        plt.plot(t, self.data_obs, '-o',mfc='none', color='k')
        plt.xlabel('time t')
        plt.ylabel('data x')
        plt.savefig('OP_data.png')