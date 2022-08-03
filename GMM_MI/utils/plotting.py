import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def choose_ax(ax, figsize=(15, 10)):
    """Select panel where to plot. Normally returns the input panel; 
    if that is None, create a new panel with the specified figsize. 
    
    Parameters
    ----------
    ax : instance of the axes.Axes class from pyplot
        The input panel.
    figsize : tuple of two integers, default=(15,10)
        The width and height of a newly created panel, in inches.
    
    Returns
    -------
    ax : instance of the axes.Axes class from pyplot
        The output panel.
    """
    _, ax1 = plt.subplots(1, 1, figsize=figsize)
    ax = ax or ax1
    return ax


def set_ticksize(ax, axis='both', which='major', labelsize=20, size=6):
    """Set the size of the ticks in a panel.   
    
    Parameters
    ----------
    ax : instance of the axes.Axes class from pyplot
        The panel whose ticks need to be set.
    axis : {'both', 'x', 'y'}, default='both'
        The axis (or axes) to which the parameters are applied.
    which : {'major', 'minor', 'both'}, default='major'
        The group of ticks to which the parameters are applied.
    labelsize : int, default=20
        Font size of the tick label.
    size : int, default=6
        Size of the ticks.
    
    Returns
    -------
    None
    """
    ax.tick_params(axis=axis, which=which, labelsize=labelsize, size=size)

    
def set_titles(ax, xlabel='', ylabel='', title='', fontsize=30):
    """Set the titles in a panel.   
    
    Parameters
    ----------
    ax : instance of the axes.Axes class from pyplot
        The panel whose titles need to be set.
    xlabel : string, default=''
        The x-label.
    ylabel : string, default=''
        The y-label.
    title : string, default=''
        The title of the panel.
    fontsize : int, default=30
        Font size of all titles and labels.
    
    Returns
    -------
    None
    """
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_xlabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    

def set_legend(ax, fontsize=25, frameon=False):
    """Set the legend in a panel.   
    
    Parameters
    ----------
    ax : instance of the axes.Axes class from pyplot
        The panel whose legend need to be set.
    fontsize : int, default=25
        Font size of the legend.
    frameon : bool, default=False
        Whether to frame the legend or not.
    
    Returns
    -------
    None
    """
    ax.legend(fontsize=fontsize, frameon=False)    

    
def calculate_ellipse_params(covariance):
    """Given a covariance matrix in 2D, calculate the orientation (angle), height and width of 
    the error ellipse representing an iso-contour of the corresponding Gaussian distribution.
    Based on https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/.
    
    Parameters
    ----------
    covariance : array-like of shape (2, 2)
        The covariance matrix of the Gaussian distribution.

    Returns
    -------
    angle : float
        Orientation of the error ellipse.
    width : float
        Width of the error ellipse.
    height : float
        Height of the error ellipse.
    """
    U, s, Vt = np.linalg.svd(covariance)
    # using arctan2 instead of arctan needed to take into account sign
    angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    width, height = 2*np.sqrt(s)
    return angle, width, height
    
    
def find_contours(contour_levels):
    """Associate the user-required sigma levels to the actual size of the error ellipse.
    The 1-sigma, 2-sigma, and/or 3-sigma contours are obtained from a chi2 table, and based on
    https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/.
    
    Parameters
    ----------
    contour_levels : list
        The user-required contours to display, in sigmas.

    Returns
    -------
    ranges : list
        The size of the error ellipse corresponding to the input contour levels.
    """
    ranges_dict = {1: np.sqrt(0.77), 2: np.sqrt(5.991), 3: np.sqrt(10.597)}
    ranges = []
    for cl in contour_levels:
        ranges.append(ranges_dict[cl])
    return ranges
        
    
def draw_ellipse(mean, covariance, weight, ax=None, contour_levels=[2],
                 weight_size=1000, alpha=None, color=None, label=None, marker='X',
                 count=0, fontsize=32, loc='lower left', frameon=False, **kwargs):
    """Draw the error ellipse corresponding to a given mean vector and Gaussian covariance matrix.
    
    Associate the user-required sigma levels to the actual size of the error ellipse.
    The 1-sigma, 2-sigma, and/or 3-sigma contours are obtained from a chi2 table, and based on
    https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/.
    
    Parameters
    ----------
    contour_levels : list
        The user-required contours to display, in sigmas.

    Returns
    -------
    None
    """
    ax = choose_ax(ax)    
    assert covariance.shape == (2, 2), "Each covariance matrix must be of shape (2, 2)!"
    # convert covariance to the parameters of the corresponding error ellipse
    angle, width, height = calculate_ellipse_params(covariance)
    # find the contour sigma levels
    contours = find_contours(contour_levels)   
    # draw the ellipse
    for contour in contours:
        ax.add_patch(Ellipse(mean, nsig*width, nsig*height, label=label if count==0 else "",
                             angle, color=color, alpha=alpha, **kwargs))
        ax.scatter(mean[0], mean[1],  marker=marker, s=weight_size*weight, color=color, alpha=alpha)
    set_legend(ax, fontsize=fontsize, frameon=frameon)   


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
        draw_ellips(mean, covariance, weight, ax=ax, alpha=alpha, fill=fill, 
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
    ax.axvline(0, ls='--', lw=lw, c='black')
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