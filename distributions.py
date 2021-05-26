import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from scipy import stats
import scipy.special as special
from scipy.stats import binom
from scipy.stats import norm
import utils_math



class binomial(object):
    '''
    The binomial distribution.
    '''

    @staticmethod
    def pdf(p, n, k):
        return binom.pmf(k, n, p)



class gamma(object):
    '''
    The Gamma distribution.
    '''

    @staticmethod
    def pdf(x, alpha, beta):
        return np.exp(gamma.logpdf(x, alpha, beta))

    @staticmethod
    def logpdf(x, alpha, beta):
        if beta > 0:
            return alpha * np.log(beta) - special.gammaln(alpha) + (alpha - 1) * np.log(x) - beta * x
        else:
            assert False, "Beta is zero"

    @staticmethod
    def cdf(x, alpha, beta):
        return special.gammainc(alpha, x * beta)

    @staticmethod
    def icdf(x, alpha, beta):
        g = stats.gamma(alpha, 0, 1.0 / beta)
        return g.ppf(x)

    @staticmethod
    def draw_samples(alpha, beta, N=1):
        return np.random.gamma(alpha, 1.0 / beta, N)




class exponential(object):
    '''
    The exponential distribution.
    '''
    @staticmethod
    def pdf(x, lamda):
        return np.exp(exponential.logpdf(x, lamda))

    @staticmethod
    def logpdf(x, lamda):
        if lamda > 0:
            return np.log(lamda) - lamda * x
        else:
            assert False, "lamda is zero"

    @staticmethod
    def draw_samples(lamda, N=1):
        return np.random.exponential(scale=1.0 / lamda, size=N)




class uniform(object):

    '''
    The 1D uniform distribution.
    '''

    @staticmethod
    def pdf(x, a=0, b=1):
        return (np.all([a <= x, x <= b], axis=0)) / (b - a)

    @staticmethod
    def logpdf(x, a=0, b=1):
        if a <= x <= b:
            return -np.log(b - a)
        return - np.inf

    @staticmethod
    def draw_samples(a=0, b=1, N=1):
        return np.random.uniform(a, b, N)




class normal(object):

    '''
    The 1D normal (or Gaussian) distribution.
    '''

    @staticmethod
    def pdf(x, mu, sigma):
        return np.exp(normal.logpdf(x, mu, sigma))

    @staticmethod
    def logpdf(x, mu, sigma):
        return -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - \
            0.5 * ((x - mu) ** 2) / (sigma ** 2)

    @staticmethod
    def cdf(x, mu, sigma):
        return stats.norm.cdf(x, mu, sigma)

    @staticmethod
    def invcdf(u, mu, sigma):
        return stats.norm.ppf(u, mu, sigma)

    @staticmethod
    def draw_samples(mu, sigma, N=1):
        return np.random.normal(mu, sigma, N)




class normal_nd(object):

    '''
    The multivariate normal (or Gaussian) distribution.
    '''

    @staticmethod
    def pdf(x, mean, cov):
        return stats.multivariate_normal.pdf(x, mean, cov, allow_singular=True)

    @staticmethod
    def logpdf(x, mean, cov):
        return stats.multivariate_normal.logpdf(x, mean, cov, allow_singular=True)

    @staticmethod
    def draw_samples(mean, cov, N=1):
        #return stats.multivariate_normal.rvs(mean, cov, N)
        return np.random.multivariate_normal(mean, cov, N)

    @staticmethod
    def fit(x):
        [n, dim] = x.shape
        mu = np.mean(x, axis=0)
        M = np.mat(x - mu)
        cov = np.matmul(M.T, M) / n
        return mu, cov





class uniform_nd(object):

    '''
    The n-dimensional uniform distribution.
    '''

    @staticmethod
    def pdf(x, p1, p2):
        float(np.all(x > p1) and np.all(x < p2)) / np.prod(p2 - p1)

    @staticmethod
    def logpdf(x, p1, p2):
        if np.all(x > p1) and np.all(x < p2):
            return -np.sum(np.log(p2 - p1))
        return -np.inf

    @staticmethod
    def draw_samples(p1, p2, N=1):
        if N == 1:
            return np.random.uniform(p1, p2)
        return np.random.uniform(p1, p2, (N, len(p1)))




class dirichlet(object):

    '''
    The dirichlet distribution.
    '''

    @staticmethod
    def pdf(x, alpha):
        return stats.dirichlet.pdf(x, alpha)


    @staticmethod
    def draw_samples(alpha1, alpha2, alpha3):
        return np.random.dirichlet([alpha1, alpha2, alpha3])




class beta(object):

    '''
    The Beta distribution.
    '''

    @staticmethod
    def pdf(x, a, b):
        return stats.beta.pdf(x, a, b)
    
    @staticmethod
    def logpdf(x, a, b):
        return np.log(stats.beta.pdf(x, a, b))

    @staticmethod
    def draw_samples(a, b, n):
        return np.random.beta(a, b, n)

    @staticmethod
    def cdf(x, a, b):
        return stats.beta.cdf(x, a, b)

    @staticmethod
    def invcdf(u, a, b):
        return stats.beta.ppf(u, a, b)




class multinomial(object):

    '''
    The Multinomial distribution.
    '''

    @staticmethod
    def draw_samples(sum, theta, N):
        return np.random.multinomial(sum, theta, size=N)




class MoG2(object):

    '''
    Mixture of Gaussian with 2 mixture components
    '''

    @staticmethod
    def pdf(x, coeff, mu1, sigma1, mu2, sigma2):
        return coeff*normal.pdf(x, mu1, sigma1) + (1-coeff)*normal.pdf(x, mu2, sigma2)

    @staticmethod
    def logpdf(x, coeff, mu1, sigma1, mu2, sigma2):
        return np.log(MoG2.pdf(x, coeff, mu1, sigma1, mu2, sigma2))

    @staticmethod
    def cdf(x, coeff, mu1, sigma1, mu2, sigma2):
        return coeff*normal.cdf(x, mu1, sigma1) + (1-coeff)*normal.cdf(x, mu2, sigma2)

    @staticmethod
    def draw_samples(coeff, mu1, sigma1, mu2, sigma2, n):
        samples = []
        for i in range(n):
            u = uniform.draw_samples(0,1,N=1)
            if u < coeff:
                x = normal.draw_samples(mu1,sigma1,N=1)[0]
            else:
                x = normal.draw_samples(mu2,sigma2,N=1)[0]
            samples.append(x)
        return np.array(samples)

    @staticmethod
    def learn_u2x_mappings(mu1, sigma1, mu2, sigma2):
        coeff = 0
        u2x_mappings = []
        while coeff <= 1:
            x = MoG2.draw_samples(coeff, mu1, sigma1, mu2, sigma2, 1000)
            u = MoG2.cdf(x, coeff,mu1, sigma1, mu2, sigma2)
            u2x_mapping = np.polyfit(u,x,deg=15)
            u2x_mappings.append(u2x_mapping)
            coeff = coeff + 0.01
        return u2x_mappings


    @staticmethod
    def invcdf(u, coeff, u2x_mappings):
        idx = int(coeff/0.01)
        if idx <0:
            idx = 0
        if idx > len(u2x_mappings)-1:
            idx = len(u2x_mappings)-1
        u2x_mapping = u2x_mappings[idx]
        poly = np.poly1d(u2x_mapping)
        return poly(u)



class MoG(object):

    '''
    Mixture of Gaussian with K mixture components
    '''

    @staticmethod
    def pdf(x, mog):
        coeffs = mog.coeffs
        means = mog.means
        sigmas = mog.sigmas

        K = len(coeffs)
        f = 0
        for k in range(K):
            f = f + coeffs[k] * normal.pdf(x, means[k], sigmas[k])
        return f

    @staticmethod
    def logpdf(x, mog):
        return np.log(MoG.pdf(x, mog)+1e-12)

    @staticmethod
    def cdf(x, mog):
        coeffs = mog.coeffs
        means = mog.means
        sigmas = mog.sigmas
    
        K = len(coeffs)
        F = 0
        for k in range(K):
            F = F + coeffs[k]*normal.cdf(x, means[k], sigmas[k])
        return F

    @staticmethod
    def draw_samples(coeffs, means, sigmas, n):
        samples = []
        cc = np.cumsum(coeffs).tolist()
        cc.insert(0,0)
        cc = np.array(cc)
        for i in range(n):
            u = uniform.draw_samples(0,1,N=1)
            if u == 1:
                idx = len(coeffs)-1
            else:
                signs = np.sign(u - cc).tolist()
                idx = signs.index(-1)-1
            x = normal.draw_samples(means[idx], sigmas[idx], N=1)[0]
            samples.append(x)
        return np.array(samples)

    @staticmethod
    def learn_u2x_mapping(coeffs, means, sigmas):
        x = MoG.draw_samples(coeffs, means, sigmas, 10000)
        u = MoG.cdf(x, coeffs, means, sigmas)
        u2x_mapping = np.polyfit(u,x,deg=15)
        return u2x_mapping

    @staticmethod
    def invcdf(u, u2x_mapping):
        poly = np.poly1d(u2x_mapping)
        return poly(u)

    @staticmethod
    def fit(X, K):
        # initialize
        n = len(X)
        mu_ = torch.randn(K, 1).float().requires_grad_(True)
        sigma_ = torch.randn(K, 1).float().requires_grad_(True)
        coeff_ = torch.randn(K, 1).float().requires_grad_(True)
        min_sigma = X.std()
        
        x = torch.tensor(X).float()
        n_val = int(0.85*n)
        idx = torch.randperm(n)
        x_train, x_val= x[idx[0:n_val]], x[idx[n_val:n]]

        # (constrained) MLE  
        T = 10000
        X = torch.tensor(X).float()
        optimizer = torch.optim.Adam([mu_, sigma_, coeff_], lr=1e-3)
        best_val_loss = math.inf
        
        def loss_func(x, mu, sigma, coeff):
            m = len(x)
            prob = torch.zeros(m, K)                  # m*K
            for k in range(K):
                normal = torch.distributions.Normal(mu[k], sigma[k])
                log_prob = normal.log_prob(x)         # m*1
                prob[:,k] = coeff[k]*log_prob.exp()   # m*1
            prob = prob.sum(dim=1)                    # m*1
            log_prob = (prob+1e-12).log().sum()       # 1*1
            return -log_prob
        
        for t in range(T):
            optimizer.zero_grad()
            
            # get parameters of MoG
            mu = mu_                                  # K*1
            sigma = F.softplus(sigma_) + min_sigma/3  # K*1
            coeff = coeff_.exp()/coeff_.exp().sum()   # K*1
            
            # optimize!
            loss = loss_func(x_train, mu, sigma, coeff)
            loss.backward()
            optimizer.step()
            
            # early stopping
            loss_eval = loss_func(x_val, mu, sigma, coeff)
            if loss_eval.item() < best_val_loss: best_val_loss = loss_eval.item()
            else: break
            
            if t%int(T/10) == 0: print('fitting Gaussian copula', 'progress=', t*1.0/T, 'loss=', loss.item())
            
        mog = MoG()
        mog.coeffs = coeff.detach().cpu().numpy()
        mog.means = mu.detach().cpu().numpy()
        mog.sigmas = sigma.detach().cpu().numpy()
        #mog.u2x_mapping = MoG.learn_u2x_mapping(mog.coeffs, mog.means, mog.sigmas)
        return mog



class kernel_distribution(object):

    '''
    A distribution defined by kernel density estimation (KDE).
    '''

    @staticmethod
    def fit(x):
        x = x.reshape(-1)
        kernel = stats.gaussian_kde(x)
        kd_object = kernel_distribution()
        kd_object.kernel = kernel
        kd_object.u2x_mapping = kernel_distribution.learn_u2x_mapping(x, kernel)
        kd_object.x2u_mapping = kernel_distribution.learn_x2u_mapping(x, kernel)
        return kd_object

    @staticmethod
    def pdf(x, kde_kernel):
        x = x.reshape(-1)
        return kde_kernel(x)

    @staticmethod
    def learn_u2x_mapping(x, kde_kernel):
        # estimate CDF by Monte Carlo integration
        delta = (x.max()-x.min())/10000
        x_array = np.linspace(x.min() + delta, x.max() - delta, 10000)
        pdf_array = kernel_distribution.pdf(x_array, kde_kernel)
        u_array = np.cumsum(pdf_array)/pdf_array.sum()
        u2x_mapping = np.polyfit(u_array,x_array,deg=10)
        return u2x_mapping

    @staticmethod
    def learn_x2u_mapping(x, kde_kernel):
        # estimate CDF by Monte Carlo integration
        delta = (x.max()-x.min())/10000
        x_array = np.linspace(x.min() + delta, x.max() - delta, 10000)
        pdf_array = kernel_distribution.pdf(x_array, kde_kernel)
        u_array = np.cumsum(pdf_array)/pdf_array.sum()
        x2u_mapping = np.polyfit(x_array,u_array,deg=10)
        return x2u_mapping

    @staticmethod
    def cdf(x, x2u_mapping):
        x = x.reshape(-1)
        poly = np.poly1d(x2u_mapping)
        return 0.01 + poly(x)*0.98

    @staticmethod
    def invcdf(u, u2x_mapping):
        u = u.reshape(-1)
        u = 0.01 + u*0.98
        poly = np.poly1d(u2x_mapping)
        return poly(u)
    
    
    
class copula:
    
    '''
    (Gaussian) Copula object
    '''
       
    def __init__(self):
        super(copula, self).__init__()
        self.gc_marginals = None
        
    @staticmethod
    def marginal_log_density(X, gc_marginals):

            '''
            :param X: n*d array
            :param gc_marginals: the learned KDE-based marginal distribution functions
            :return: n*1 array that contains the marginal density of GC
            '''
            logpdf_marginals = 0
            [n, dim] = X.shape
            for k in range(dim):
                gc_marginal = gc_marginals[k]
                #v = kernel_distribution.pdf(X[:, k], gc_marginal.kernel)             
                v = MoG.pdf(X[:,k], gc_marginal)
                logpdf_marginals = logpdf_marginals + np.log(v+1e-15)
            return logpdf_marginals

    @staticmethod
    def copula_density(Z, V):

            '''
            :param Z: n*d array that contains the latent variable z in Gaussian copula (i.e  Phi(z_i) = F(x_i) )
            :param V: covariance matrix of Gaussian copula
            :return: n*1 array that contains c(z^(1)), c(z^(2)), ... c(z^(n))
            '''

            (n, dim) = Z.shape
            A = normal_nd.pdf(Z, np.zeros([dim, 1]).reshape(-1), V)
            B = np.prod(normal.pdf(Z, 0, 1), axis=1)
            return A/(B+1e-20)

    @staticmethod
    def cdf_per_dim(X):

            '''
            :param X: n*d array of data
            :return: a data structure that contains (quantile, CDF) values
            '''

            cdf_per_dimension = []
            dim = X.shape[1]
            for k in range(dim):
                quantiles, CDF = utils_math.ecdf(np.asarray(X[:, k]))
                cdf_per_dimension.append((quantiles, CDF))
            return cdf_per_dimension

    @staticmethod
    def X2U(X, cdf_per_dimension):
            '''
            :param X: n*d ndarray/matrix
            :return: U, n*d ndarray/matrix
            '''

            (n, dim) = X.shape
            U = []
            for i in range(n):
                x = np.atleast_2d(X[i, 0:])
                u = []
                for k in range(dim):
                    quantile = cdf_per_dimension[k][0]
                    cdf = cdf_per_dimension[k][1]
                    u.append(utils_math.get_ecdf_val(quantile, cdf, x[0, k]))
                U.append(np.array(u))

            # step 2. filter extreme value in yetas
            U = np.mat(U)
            return np.array(U)

    @staticmethod
    def X2Z(X, cdf_per_dimension):

            '''
            :param X: n*d ndarray/matrix
            :return: Z, n*d ndarray/matrix
            '''

            # step 1. then convert CDF, u -> z
            (n, dim) = X.shape
            Z = []
            for i in range(n):
                x = np.atleast_2d(X[i, 0:])
                z = []
                for k in range(dim):
                    quantile = cdf_per_dimension[k][0]
                    cdf = cdf_per_dimension[k][1]
                    z.append(stats.norm.ppf(utils_math.get_ecdf_val(quantile, cdf, x[0, k])))
                Z.append(np.array(z))

            # step 2. filter extreme value in yetas
            Z = np.mat(Z)
            Z[Z == np.inf] = 0.0
            Z[Z == -np.inf] = 0.0

            return np.array(Z)

    @staticmethod
    def Z2X(Z, cdf_per_dimension):

            '''
            :param Z: n*d ndarray/matrix
            :return: X, n*d ndarray/matrix
            '''

            X = []
            n = len(Z)
            dim = Z[0].size
            for i in range(n):
                z = Z[i]
                x = []
                z = np.atleast_1d(z)
                for k in range(dim):
                    quantile = cdf_per_dimension[k][0]
                    cdf = cdf_per_dimension[k][1]
                    x.append(utils_math.inverse_ecdf(quantile, cdf, stats.norm.cdf(z[k])))
                X.append(np.array(x))
            return np.array(X)

    def fit(self, X):
        # 1. learn the marginal distribution
        gc_marginals = []
        [n, dim] = X.shape
        for k in range(dim):
            #gc_marginal = kernel_distribution.fit(X[:, k])
            gc_marginal = MoG.fit(X[:, k], K=3)
            gc_marginals.append(gc_marginal)
        gc_cdf_mapping = copula.cdf_per_dim(X)

        # 2. learn the correlation structure
        U = np.zeros((n, dim))
        for k in range(dim):
            gc_marginal = gc_marginals[k]
            U[:, k] = MoG.cdf(X[:, k], gc_marginal)
        Z = norm.ppf(U)
        gc_cov = np.mat(Z).T * np.mat(Z) / n
        
        self.gc_cdf_mapping = gc_cdf_mapping
        self.gc_marginals = gc_marginals
        self.gc_cov = gc_cov

    def logpdf(self, x):
        x = np.atleast_2d(x)
        [n, dim] = x.shape

        # 1. compute marginal pdf
        logpdf_marginals = copula.marginal_log_density(x, self.gc_marginals)
        
        # 2. compute copula density
        u = np.zeros((n, dim))
        for k in range(dim):
            gc_marginal = self.gc_marginals[k]
            u[:, k] = MoG.cdf(x[:, k], gc_marginal)
        z = norm.ppf(u)        
        c = copula.copula_density(z, self.gc_cov)
        logpdf_c = np.log(c)

        # copula density * marginal pdf = likelihood
        return logpdf_c + logpdf_marginals

    def sample(self):
        z = normal_nd.draw_samples(mean=np.zeros(self.gc_cov.shape[0]), cov=self.gc_cov, N=1)
        z = np.atleast_2d(z)
        x = copula.Z2X(z, self.gc_cdf_mapping)
        return x