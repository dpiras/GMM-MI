import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, weight, ax=None, alpha=None, color=None, label=None, count=0, **kwargs):
    """
    TODO
    Draw an ellipse with a given position and covariance
    """
    ax = ax or plt.gca()
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
        #print(s)
        
    # draw the ellipse
    # These should be 68% and 95% contours, from a chi2 table
    ranges = [np.sqrt(5.991)] #[np.sqrt(0.77), np.sqrt(5.991)]
    for nsig in ranges:
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, color=color, alpha=alpha,  label=label if count==0 else "", **kwargs))
    #ax.scatter(position[0], position[1],  marker='X', s=0*weight, color=color, alpha=alpha, label=label if count==0 else "")
    ax.legend(fontsize=32, frameon=False, loc='lower left')
    
        
def plot_contours(gmm, ax=None, color='salmon', ls='--', label='True', scatter='True'):
    """
    TODO
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))

    ax = ax or ax1    
    means, covariances, weights = gmm.means_, gmm.covariances_, gmm.weights_
    w_factor = 1 # just for the plot
    count=0
    for mean, covariance, weight in zip(means, covariances, weights):
        draw_ellipse(mean, covariance, weight, ax=ax, alpha=0.8, fill=False, color=color,ls=ls, linewidth=4, label=label, count=count)
        count += 1
        
    if scatter:
        X = gmm.sample(1e4)[0]
        ax.scatter(X[:, 0], X[:, 1])
        
    ax.tick_params(axis='both', which='major', labelsize=20, size=6)
    ax.set_xlabel('X1', fontsize=30)
    ax.set_ylabel('X2', fontsize=30)

def plot_MI_values(MI_estimates, bins=20, alpha=1, legendsize=25, color='salmon', lw=3, title=None):
    fig, (ax) = plt.subplots(1, 1, figsize=(15, 10))
    trials = len(MI_estimates)

    # calculate significance
    mean_error = np.std(MI_estimates)/np.sqrt(trials)
    significance = np.mean(MI_estimates) / mean_error
    
    # and plot
    ax.hist(MI_estimates.flatten(), alpha=alpha, bins=bins, 
             label=f'MI values, {significance:.1f}$\sigma$,\n{np.mean(MI_estimates):.4f}$\pm${mean_error:.4f}', 
             color=color, histtype='step', lw=lw)

    ax.axvline(0, ls='--', lw=lw, c='k')
    ax.tick_params(axis='both', which='major', labelsize=20, size=6)
    ax.legend(fontsize=legendsize, frameon=False)
    ax.set_xlabel('MI [nats]', fontsize=30)
    ax.set_xlabel('Counts', fontsize=30)

    # change title accordingly
    ax.set_title(title, fontsize=30)