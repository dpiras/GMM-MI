import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def choose_ax(ax, figsize=(15, 10)):
    _, ax1 = plt.subplots(1, 1, figsize=figsize)
    ax = ax or ax1
    return ax

def set_ticksize(ax, axis='both', which='major', labelsize=20, size=6):
    ax.tick_params(axis=axis, which=which, labelsize=labelsize, size=size)

def set_titles(ax, xlabel='', ylabel='', title='', fontsize=30):
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_xlabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    
def set_legend(ax, fontsize=25, frameon=False):
    ax.legend(fontsize=fontsize, frameon=False)    

def calculate_ellipse_params(covariance):
    U, s, Vt = np.linalg.svd(covariance)
    angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    width, height = 2*np.sqrt(s)
    return angle, width, height
    
def find_contours(contour_levels):
    # these should be 68%, 95%, and/or 99.5% contours, from a chi2 table
    ranges_dict = {1: np.sqrt(0.77), 2: np.sqrt(5.991), 3: np.sqrt(10.597)}
    ranges = []
    for cl in contour_levels:
        ranges.append(ranges_dict[cl])
    return ranges
        
def draw_ellipses(mean, covariance, weight, ax=None, contour_levels=[2],
                 weight_size=1000, alpha=None, color=None, label=None, marker='X',
                 count=0, fontsize=32, loc='lower left', frameon=False, **kwargs):
    """
    TODO
    Draw an ellipse with a given mean and covariance
    """
    ax = choose_ax(ax)    
    # check covariance.shape == (2, 2)
    # convert covariance to principal axes
    angle, width, height = calculate_ellipse_params(covariance)
        
    # find the contour sigma levels
    contours = find_contours(contour_levels)
    
    # draw the ellipse
    for contour in contours:
        ax.add_patch(Ellipse(mean, nsig*width, nsig*height, label=label if count==0 else "",
                             angle, color=color, alpha=alpha, **kwargs))
        ax.scatter(mean[0], mean[1],  marker=marker, s=weight_size*weight, color=color, alpha=alpha)
    ax.legend(fontsize=fontsize, frameon=frameon, loc=loc)
    
def plot_samples(gmm, N=1e4):
    X = gmm.sample(N)[0]
    ax.scatter(X[:, 0], X[:, 1])
    
def plot_contours(gmm, ax=None, color='salmon', ls='--', alpha=0.8, 
                  linewidth=4, fill=False, label='True', scatter='True', 
                  N=1e4, xlabel='X1', ylabel='X2'):
    """
    TODO
    """
    ax = choose_ax(ax, figsize=(10, 10))
    means, covariances, weights = gmm.means_, gmm.covariances_, gmm.weights_
    for count, (mean, covariance, weight) in enumerate(zip(means, covariances, weights)):
        draw_ellipses(mean, covariance, weight, ax=ax, alpha=alpha, fill=fill, 
                     color=color,ls=ls, linewidth=linewidth, label=label, count=count)  
    if scatter:
        plot_scatter(gmm, N=N):
        
    set_ticksize(ax)
    set_titles(ax, xlabel=xlabel, ylabel=ylabel)


def calculate_distribution_and_significance(MI_estimates):
    trials_number = len(MI_estimates)
    # calculate significance
    mean_value = np.mean(MI_estimates)
    mean_error = np.std(MI_estimates)/np.sqrt(trials_number)
    significance = np.mean(MI_estimates) / mean_error
    return mean_value, mean_error, significance


def plot_MI_values(MI_estimates, ax=None, bins=20, alpha=1, title='',
                   legendsize=25, color='salmon', lw=3, histtype='step'):
    
    ax = choose_ax(ax)
    mean_value, mean_error, significance = calculate_distribution_and_significance(MI_estimates)
    ax.hist(MI_estimates.flatten(), alpha=alpha, bins=bins, color=color, histtype=histtype, lw=lw,
            label=f'MI values, {significance:.1f}$\sigma$,\n{mean_value:.4f}$\pm${mean_error:.4f}' 
            )
    ax.axvline(0, ls='--', lw=lw, c='k')
    set_ticksize(ax)
    set_titles(ax, xlabel='MI [nats]', ylabel='Counts', title=title)
    set_legend(ax, fontsize=legendsize)

def select_labels(i):
    if i == 0:
        label1 = 'Train'
        label2 = 'Valid'
    else:
        label1 = '' 
        label2 = ''
    return label1, label2

def plot_loss_curves_individual(loss_curves, i, ax):
    current_train_loss = loss_curves[2*i]
    current_val_loss = loss_curves[2*i+1]
    x_v = np.arange(len(current_train_loss))
    label1, label2 = select_labels(i)

    ax.plot(x_v, current_train_loss, label=label1, lw=3, ls='-', color='grey')
    ax.plot(x_v, current_val_loss, label=label2, lw=3, ls='-', color='salmon')
    
def plot_loss_curves(loss_curves, n_inits, n_folds, figsize=(20, 30), 
                     legendsize=18, frameon=False, loc='lower right'):
    fig, axes = plt.subplots(n_inits, n_folds, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i in range(n_inits*n_folds):
        ax = axes[i]
        plot_loss_curves_individual(loss_curves, i, ax)
        if i == 0:
            set_legend(ax, fontsize=legendsize, frameon=frameon, loc=loc)
        set_ticksize(ax)        
        if i >= (n_inits-1)*n_folds:
            set_titles(ax, xlabel='Iteration', fontsize=20)
        if i % n_folds == 0:
            set_titles(ax, ylabel='logL', fontsize=20)
    fig.subplots_adjust(wspace=0.0, hspace=0.0)