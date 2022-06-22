import numpy as np
import torch
import scipy.stats as stats
import scipy.spatial as spatial
import scipy.integrate as integrate



# ||s - s_obs||
def eculidean_dist(s, s_obs):
    s, s_obs = s.flatten(), s_obs.flatten()
    delta = (s-s_obs)
    return np.linalg.norm(delta)



# max(|s-s_obs|)
def chebyshev_dist(s, s_obs):
    delta = np.abs(s-s_obs)
    return delta.max()



# Sample-based KL. See [1][2]: KL as data discrepancy in ABC.
def quasi_KL(samples_P, samples_Q):
    (n, dim) = samples_P.shape
    (m, dim) = samples_Q.shape

    M_PQ = spatial.distance_matrix(x=np.mat(samples_P), y=np.mat(samples_Q), p=2)
    M_PP = spatial.distance_matrix(x=np.mat(samples_P), y=np.mat(samples_P), p=2)

    M_PQ = M_PQ.min(axis=1)
    M_PP.sort(axis=1)

    min_x2y = np.array(M_PQ)+0e-8
    min_x2x = np.array(M_PP[:, 1])+0e-8

    ret = dim/n * np.log(min_x2y/min_x2x).sum() + np.log(m/(n-1))
    return ret



# KL-divergence
def KL(samples_P, samples_Q):

    # note: estimated by KDE, might be inaccurate in dim>4 cases

    (n, dim) = samples_P.shape
    (m, dim) = samples_Q.shape

    kernel_p = stats.gaussian_kde(samples_P.T)
    kernel_q = stats.gaussian_kde(samples_Q.T)

    samples = np.vstack((samples_P, samples_Q))
    min_values = samples.min(axis=0)
    max_values = samples.max(axis=0)

    N = 30
    ranges = []
    for k in range(dim):
        ranges.append(np.array(np.linspace(min_values[k], max_values[k], N)))
    R = np.array(np.meshgrid(*ranges)).T.reshape(-1, dim)
    R = R.T

    prob_p = kernel_p(R)
    prob_q = kernel_q(R)

    # Monte carlo integration for KL computation
    KL = 0

    prob_p = prob_p/sum(prob_p)
    prob_q = prob_q/sum(prob_q)
    for i in range(N**dim):
        if prob_p[i] < 1e-10:
            KL += 0
        else:
            KL += prob_p[i] * np.log(prob_p[i]/prob_q[i])
    return KL



# KL-divergence
def KL2(p, q, samples_P, samples_Q):

    # Monte Carlo approximation of the JSD value

    # Evaluate p(x), q(x) at these points
    [n, dim] = samples_P.shape
    samples = np.vstack((samples_P, samples_Q))
    min_values = samples.min(axis=0)
    max_values = samples.max(axis=0)
    N = 150
    ranges = []
    for k in range(dim):
        ranges.append(np.array(np.linspace(min_values[k], max_values[k], N)))
    R = np.array(np.meshgrid(*ranges)).T.reshape(-1, dim)

    prob_p = p(R)
    prob_q = q(R)

    # Monte carlo integration for KL computation
    KL = 0
    prob_p = prob_p/sum(prob_p)
    prob_q = prob_q/sum(prob_q)
    for i in range(N**dim):
        if prob_p[i] < 1e-10:
            KL += 0
        else:
            KL += prob_p[i]*np.log(prob_p[i]/prob_q[i])
    return KL



# Maximum Mean Discrepancy
def MMD(samples_P, samples_Q):
    
    # > sample-based MMD calculation
    
    (n, dim) = samples_P.shape
    (m, dim) = samples_Q.shape

    M_PP = spatial.distance_matrix(x=np.mat(samples_P), y=np.mat(samples_P), p=2)
    M_QQ = spatial.distance_matrix(x=np.mat(samples_Q), y=np.mat(samples_Q), p=2)
    M_PQ = spatial.distance_matrix(x=np.mat(samples_P), y=np.mat(samples_Q), p=2)

    bandwidth = np.median(M_PP.reshape(-1))

    M_PP = np.exp(- (M_PP*M_PP) / (2.0 * bandwidth**2) )
    M_QQ = np.exp(- (M_QQ*M_QQ) / (2.0 * bandwidth**2) )
    M_PQ = np.exp(- (M_PQ*M_PQ) / (2.0 * bandwidth**2) )

    A = M_PP.sum().sum()/(n*(n-1))
    B = M_QQ.sum().sum()/(m*(m-1))
    C = M_PQ.reshape(-1).mean()

    return (A+B-2*C)**0.5





# Jensen-Shannon divergence
def JSD(log_p, log_q, samples_P, samples_Q, N_grid=30):

    # > Monte Carlo approximation of the JSD value

    # Evaluate p(x), q(x) at these points
    [n, dim] = samples_P.shape
    (m, dim) = samples_Q.shape

    samples = np.vstack((samples_P, samples_Q))
    min_values = samples.min(axis=0)
    max_values = samples.max(axis=0)
    N = N_grid
    ranges = []
    for k in range(dim):
        ranges.append(np.array(np.linspace(min_values[k], max_values[k], N)))
    R = np.array(np.meshgrid(*ranges)).T.reshape(-1, dim)

    prob_p = np.zeros((N**dim, 1))
    prob_q = np.zeros((N**dim, 1))
    for i in range(N**dim):
        sample = R[i]
        prob_p[i] = np.exp(log_p(sample))
        prob_q[i] = np.exp(log_q(sample))
    
    # Riemann integration for KL computation
    KL_PM = 0
    KL_QM = 0
    prob_p = prob_p/sum(prob_p)
    prob_q = prob_q/sum(prob_q)
    prob_pq = (prob_p + prob_q)/2
    for i in range(N**dim):
        if prob_p[i] < 1e-20:
            KL_PM += 0
        else:
            KL_PM += prob_p[i] * np.log(prob_p[i]/prob_pq[i])

        if prob_q[i] < 1e-20:
            KL_QM += 0
        else:
            KL_QM += prob_q[i] * np.log(prob_q[i]/prob_pq[i])

    return KL_PM/2 + KL_QM/2



# Jensen-Shannon divergence
def JSD2(log_P, log_Q, log_P2, samples_P2):

    # > Monte Carlo approximation of the JSD value using importance sampling
    
    [n, dim] = samples_P2.shape
    KL_PM = 0
    KL_QM = 0
    for i in range(n):
        x = samples_P2[i]
        log_p = log_P(x)
        log_q = log_Q(x)
        log_pq = np.log((np.exp(log_p) + np.exp(log_q))/2.0)
        log_p2 = log_P2(x)
        f = log_p - log_pq
        g = log_q - log_pq
        w_f = np.exp(log_p - log_p2)
        w_g = np.exp(log_q - log_p2)
        KL_PM += (w_f*f)/n
        KL_QM += (w_g*g)/n
    return KL_PM/2 + KL_QM/2





# # Maximum mean discrepancy in torch
# def MMD_torch(x_de, x_nu, σs=[]):
#     # define some functions
#     def gaussian_gramian(esq, σ):
#         return torch.div(-esq, 2 * σ**2).exp()
#     def prepare(x_de, x_nu):
#         dsq_dede = torch.pow(torch.cdist(x_de, x_de), 2)
#         dsq_denu = torch.pow(torch.cdist(x_de, x_nu), 2)
#         dsq_nunu = torch.pow(torch.cdist(x_nu, x_nu), 2)
#         return dsq_dede, dsq_denu, dsq_nunu
    
#     dsq_dede, dsq_denu, dsq_nunu = prepare(x_de, x_nu)
    
#     # determine σs
#     if len(σs) == 0:
#         # A heuristic is to use the median of pairwise distances as σ, suggested by Sugiyama's book
#         sigma = torch.sqrt(
#             torch.median(
#                 torch.cat([dsq_dede.view(-1), dsq_denu.view(-1), dsq_nunu.view(-1)], dim=0)
#             )
#         ).item()
#         # Use [sigma / 5, sigma / 3, sigma, sigma * 3, sigma * 5] if nothing provided
#         σs = [sigma, sigma * 0.333, sigma * 0.2, sigma / 0.2, sigma / 0.333]

#     # now compute MMD
#     is_first = True
#     mmdsq = None
#     for σ in σs:
#         Kdede = gaussian_gramian(dsq_dede, σ)
#         Kdenu = gaussian_gramian(dsq_denu, σ)
#         Knunu = gaussian_gramian(dsq_nunu, σ)
#         if is_first:
#             mmdsq = torch.mean(Kdede) - 2 * torch.mean(Kdenu) + torch.mean(Knunu)
#             is_first = False
#         else:
#             mmdsq += torch.mean(Kdede) - 2 * torch.mean(Kdenu) + torch.mean(Knunu)
#             is_first = False
#     mmd = torch.sqrt(torch.relu(mmdsq))
#     return mmd