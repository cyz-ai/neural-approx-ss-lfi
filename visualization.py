import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import time
import utils_math, utils_os
import discrepancy



def plot_contour(X, Y, p_XY, region, title, xlabel, ylabel):
    plt.contour(X, Y, p_XY, 10, cmap='jet', linewidths=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    xmin = region[0, 0]
    xmax = region[0, 1]
    ymin = region[1, 0]
    ymax = region[1, 1]
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))

def plot_axline(x,y):
    plt.axvline(x=x,color='k', linewidth=0.75)
    plt.axhline(y=y,color='k', linewidth=0.75)
    
def plot_likelihood(samples, log_likelihood_function, dimensions=(0,1)): 
    # Compute log-likelihood values
    n, d = samples.shape
    if d == 1:
        X, P = utils_math.log_likelihood_1D(samples, log_likelihood_function)
        plt.figure(time.time()*100, figsize=(5, 5))
        plt.plot(X,P)
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\log p(\theta|x_o)$')
        return
    elif d == 2:
        X, Y, P = utils_math.log_likelihood_2D(samples, log_likelihood_function)
    else:
        X, Y, P = utils_math.log_likelihood_3D(samples, log_likelihood_function, dimensions)
        
    # Determine the visualize region
    visualize_samples = samples[:, dimensions]
    (mean1, std1) = visualize_samples[:,0].mean(), visualize_samples[:,0].std()
    (mean2, std2) = visualize_samples[:,1].mean(), visualize_samples[:,1].std()
    print('mean-parma1 = ', mean1, '     mean-param2 = ', mean2)
    R = np.array([[mean1-3.0*std1, mean1+3.0*std1], [mean2-3.0*std2, mean2+3.0*std2]])
    
    # Visualize contour
    fig = plt.figure(time.time()*100, figsize=(5, 5))
    #fig.set_tight_layout(True)
    #plt.axis('off')
    plot_contour(X, Y, P, R, '', r'$\theta_{}$'.format(dimensions[0]), r'$\theta_{}$'.format(dimensions[1]))


def plot_samples(samples, dimensions=(0,1)):
    visualize_samples = samples[:, dimensions]
    plt.figure(time.time()*100, figsize=(5, 4))
    plt.scatter(visualize_samples[:,0], visualize_samples[:,1], s=8, edgecolors='k', marker='o', facecolors='none')
    
    
def compare_contours(samples, true_likelihood, approx_likelihood, dimensions=(0,1)):
    # Compute log-likelihood values
    n, d = samples.shape
    if d == 2:
        X, Y, P1 = utils_math.log_likelihood_2D(samples, true_likelihood)
        X, Y, P2 = utils_math.log_likelihood_2D(samples, approx_likelihood)
    else:
        X, Y, P1 = utils_math.log_likelihood_3D(samples, true_likelihood, dimensions)
        X, Y, P2 = utils_math.log_likelihood_3D(samples, approx_likelihood, dimensions)
        
    # Determine the visualize region
    visualize_samples = samples[:, dimensions]
    (mean1, std1) = visualize_samples[:,0].mean(), visualize_samples[:,0].std()
    (mean2, std2) = visualize_samples[:,1].mean(), visualize_samples[:,1].std()
    print('mean-parma1 = ', mean1, '     mean-param2 = ', mean2)
    R = np.array([[mean1-3.0*std1, mean1+3.0*std1], [mean2-3.0*std2, mean2+3.0*std2]])
    
    # Visualize contour
    plt.figure(time.time()*100, figsize=(5, 4))
    C1 = plt.contour(X, Y, P1, 10, colors='k', linewidths=0.75)
    C2 = plt.contour(X, Y, P2, 10, colors='r', linewidths=0.75, linestyles='dashed', alpha=0.85)
    C1.collections[0].set_label('true posterior')
    C2.collections[0].set_label('approx posterior')
    plt.legend(bbox_to_anchor=(-0.00,1.02,1.00,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2)
    plt.title('')
    plt.xlabel(r'$\theta_{}$'.format(dimensions[0]))
    plt.ylabel(r'$\theta_{}$'.format(dimensions[1]))
    xmin = R[0, 0]
    xmax = R[0, 1]
    ymin = R[1, 0]
    ymax = R[1, 1]
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))
    plt.savefig('contours_compare.png')
    
    
