import numpy as np
import scipy.stats as stats


def ecdf(sample):
    # compute cdf = rank(x)/n
    sample = np.atleast_1d(sample)
    quantiles, counts = np.unique(sample, return_counts=True)
    cdf = np.cumsum(counts).astype(np.double) / sample.size
    return quantiles, cdf


def get_ecdf_val(quantiles, cdf, x):
    # first find the rank of x in quantiles
    sign_list = np.sign(quantiles - x).tolist()
    if 1 in sign_list:
        index = sign_list.index(1)-1
        if index < 0:
            return 0.0
        else:
            return cdf[index]
    else:
        return 1.0


def inverse_ecdf(quantiles, cdf, u):
    if u == 1:
        return quantiles[-1]
    elif u == 0:
        return quantiles[0]
    elif u < 0 or u > 1:
        raise Exception('the input probability must be between 0 and 1')
    else:
        sign_list = np.sign(cdf - u).tolist()
        if 1 not in sign_list:
            return quantiles[-1]
        right_index = sign_list.index(1)
        left_index = right_index - 1
        left_abs = round(np.abs(cdf[left_index] - u), 3)
        right_abs = round(np.abs(cdf[right_index] - u), 3)
        if left_abs < right_abs:
            return quantiles[left_index]
        elif right_abs < left_abs:
            return quantiles[right_index]
        else:
            return (quantiles[left_index] + quantiles[right_index]) / 2.0

        
def kde_1D(samples):
    data = samples.transpose()
    kde_kernels = stats.gaussian_kde(data)
    X = np.linspace(data.min(), data.max(), 2000)
    Y = kde_kernels(X)
    return (X, Y)


def kde_2D(samples):
    data = samples.transpose()
    kde_kernel = stats.gaussian_kde(data)

    xmin = data[0, 0:].min()
    xmax = data[0, 0:].max()
    ymin = data[1, 0:].min()
    ymax = data[1, 0:].max()

    X, Y = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kde_kernel(positions).T, X.shape)
    return (X,Y,Z)


def log_likelihood_1D(samples, log_likelihood_function):
    xmin = samples.flatten().min()
    xmax = samples.flatten().max()
    X = np.linspace(xmin, xmax, 1000)
    Z = []
    for x in X:
        theta = np.atleast_2d(x)
        log_pdf = log_likelihood_function(theta)
        Z.append(log_pdf)
    return (X, Z)
    
def log_likelihood_2D(samples, log_likelihood_function):
    data = samples.transpose()
    xmin = data[0, 0:].min()
    xmax = data[0, 0:].max()
    ymin = data[1, 0:].min()
    ymax = data[1, 0:].max()

    X, Y = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
    Z = np.zeros(X.shape)
    for i in range(50):
        for j in range(50):
            x = X[i,j]
            y = Y[i,j]
            theta = [x,y]
            logpdf = log_likelihood_function(theta)
            Z[i,j] = np.exp(logpdf)
    return (X, Y, Z)


def log_likelihood_3D(samples, log_likelihood_function, dimensions):
    data = samples.transpose()
    xmin = data[0, 0:].min()
    xmax = data[0, 0:].max()
    ymin = data[1, 0:].min()
    ymax = data[1, 0:].max()
    zmin = data[2, 0:].min()
    zmax = data[2, 0:].max()

    # MC approximation    
    X, Y, Z = np.mgrid[xmin:xmax:30j, ymin:ymax:30j, zmin:zmax:30j]
    if dimensions == (0,1):
        XX, YY = np.mgrid[xmin:xmax:30j, ymin:ymax:30j]
        F = np.zeros(XX.shape)
        for i in range(30):
            for j in range(30):
                for k in range(30):
                    x = X[i,j,k]
                    y = Y[i,j,k]
                    z = Z[i,j,k]
                    theta = [x,y,z]
                    logpdf = log_likelihood_function(theta)
                    f_ijk = np.exp(logpdf)
                    F[i,j] = F[i,j] + f_ijk
        return (XX,YY,F)

    if dimensions == (0,2):
        XX, ZZ = np.mgrid[xmin:xmax:30j, zmin:zmax:30j]
        F = np.zeros(XX.shape)
        for i in range(30):
            for j in range(30):
                for k in range(30):
                    x = X[i,j,k]
                    y = Y[i,j,k]
                    z = Z[i,j,k]
                    theta = [x,y,z]
                    logpdf = log_likelihood_function(theta)
                    f_ijk = np.exp(logpdf)
                    F[i,k] = F[i,k] + f_ijk
        return (XX,ZZ,F)

    if dimensions == (1,2):
        YY, ZZ = np.mgrid[ymin:ymax:30j, zmin:zmax:30j]
        F = np.zeros(YY.shape)
        for i in range(30):
            for j in range(30):
                for k in range(30):
                    x = X[i,j,k]
                    y = Y[i,j,k]
                    z = Z[i,j,k]
                    theta = [x,y,z]
                    logpdf = log_likelihood_function(theta)
                    f_ijk = np.exp(logpdf)
                    F[j,k] = F[j,k] + f_ijk
        return (YY,ZZ,F)


def MAD(x):
    m = np.median(x, axis=0)
    xx = np.abs(x-m)
    return np.median(xx, axis=0)


def is_pos_def(M):
    return np.all(np.linalg.eigvals(M) > 0)


def discrete_sample(p, n_samples=1):
    """
    Samples from a discrete distribution.
    :param p: a distribution with N elements
    :param n_samples: number of samples
    :return: vector of samples
    """

    # check distribution
    #assert isdistribution(p), 'Probabilities must be non-negative and sum to one.'

    # cumulative distribution
    c = np.cumsum(p[:-1])[np.newaxis, :]

    # get the samples
    r = np.random.rand(n_samples, 1)
    return np.sum((r > c).astype(int), axis=1)