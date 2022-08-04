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
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
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
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    

def set_legend(ax, fontsize=25, frameon=False, loc='best'):
    """Set the legend in a panel.   
    
    Parameters
    ----------
    ax : instance of the axes.Axes class from pyplot
        The panel whose legend need to be set.
    fontsize : int, default=25
        Font size of the legend.
    frameon : bool, default=False
        Whether to frame the legend or not.
    loc : string, default='best'
        Position of the legend.     
    
    Returns
    -------
    None
    """
    ax.legend(fontsize=fontsize, frameon=frameon, loc=loc)    

    
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
        Only 1, 2 and 3 are accepted in this list.

    Returns
    -------
    ranges : list
        The size of the error ellipse corresponding to the input contour levels.
    """
    if any(cl not in [1,2,3] for cl in contour_levels):
        raise ValueError(f'Contour levels must be either 1, 2 and 3; current contour levels are {contour_levels}.')
    assert len(contour_levels) <= 3, 'Length of contour levels cannot exceed 3'
    ranges_dict = {1: np.sqrt(0.77), 2: np.sqrt(5.991), 3: np.sqrt(10.597)}
    ranges = []
    for cl in contour_levels:
        ranges.append(ranges_dict[cl])
    return ranges
        
    
def draw_ellipse(mean, covariance, weight, ax=None, contour_levels=[2],
                 weight_size=1000, alpha=None, color=None, label=None, marker='X',
                 component_count=0, legendsize=32, loc='lower left', frameon=False, **kwargs):
    """Draw the error ellipse corresponding to a given mean vector and Gaussian covariance matrix.
    Since this is don in the context of Gaussian mixture models (GMMs), the weight is also indicated 
    with a marker whose size is proportional to the weight value. Based on
    https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/.
    
    Parameters
    ----------
    mean : array-like of shape (2)
        The mean of the Gaussian distribution.
    covariance : array-like of shape (2, 2)
        The covariance matrix of the Gaussian distribution.
    weight : float
        The weight associated with this GMM component.
    ax : instance of the axes.Axes class from pyplot, default='None'
        The panel where to draw the ellipse.    
    contour_levels : list of integers, default=[2]
        The sigma levels whose contour are going to be drawn. 
        The integers can be only 1, 2 and 3.
    weight_size : float, default=1000
        The multiplier for the marker size (also proportional to the weight).   
    alpha : float, default=None
        Transparency, between 0 and 1.    
    color : 'string', default=None
        Color of the ellipse.
    marker : string, default='X'
        Marker type for the centroid of the ellipse.
    label : string, default=None
        Label associated to the ellipse.
    component_count : int, default=0
        A counter so that the label is printed only for the first component.
    legendsize : int, default=32
        Font size of the legend.
    loc : string, default='lower left'
        Legend position.
    frameon : bool, default=False
        Whether to frame the legend or not.
        
    Returns
    -------
    None
    """
    ax = choose_ax(ax)
    assert covariance.shape == (2, 2), 'Each covariance matrix must be of shape (2, 2)!'
    # convert covariance to the parameters of the corresponding error ellipse
    angle, width, height = calculate_ellipse_params(covariance)
    # find the contour sigma levels
    contours = find_contours(contour_levels)   
    # draw the ellipse
    for contour in contours:
        ax.add_patch(Ellipse(mean, contour*width, contour*height, angle, color=color, 
                             alpha=alpha, label=label if component_count==0 else "", **kwargs))
        ax.scatter(mean[0], mean[1],  marker=marker, color=color,
                   s=weight_size*weight, alpha=alpha)
    set_legend(ax, fontsize=legendsize, frameon=frameon)


def plot_samples(gmm, ax, N=1e4):
    """Produce a scatter plot of the samples of a Gaussian mixture model (GMM).
    
    Parameters
    ----------
    gmm : instance of the GMM class
        The GMM whose samples are going to be displayed.
    ax : instance of the axes.Axes class from pyplot
        The panel where to plot the samples.     
    N : int, default=1e4
        The number of samples to plot.

    Returns
    -------
    None
    """
    X = gmm.sample(N)[0]
    ax.scatter(X[:, 0], X[:, 1])
    
    
def plot_gmm_contours(gmm, ax=None, color='salmon', ls='--', alpha=0.8, 
                  linewidth=4, fill=False, label='', scatter=True, 
                  N=1e4, xlabel='X1', ylabel='X2'):
    """Draw the contour ellipses corresponding to a given Gaussian mixture model (GMM),
    including a possible scatte plot of a required number of samples from the GMM.
    
    Parameters
    ----------
    gmm : instance of the GMM class
        The GMM whose contours are going to be displayed.
    ax : instance of the axes.Axes class from pyplot
        The panel where to plot the samples.   
    color : string, default='salmon'
        The color of the GMM contours.
    ls : string, default='--'
        The line style of the contours.
    alpha : float, default=0.8
        Transparency of the contour lines; must be between 0 and 1.
    linewidth : int, default=4
        The width of the contour lines.
    fill : bool, default=False
        Whether to fill the contours or not.
    label : string, default=''
        The legend label to associate to the contours.
    scatter : bool, default=True
        Whether to also display a scatter plot with samples of the GMM.
    N : int, default=1e4
        The number of samples to plot in the scatter plot.
    xlabel : string, default='X1'
        Label of the x-axis.
    ylabel : string, default='X2'
        Label of the y-axis. 
        
    Returns
    -------
    None
    """
    ax = choose_ax(ax, figsize=(10, 10))
    means, covariances, weights = gmm.means_, gmm.covariances_, gmm.weights_
    for component_count, (mean, covariance, weight) in enumerate(zip(means, covariances, weights)):
        draw_ellipse(mean, covariance, weight, ax=ax, alpha=alpha, fill=fill, 
                     color=color,ls=ls, linewidth=linewidth, label=label, component_count=component_count)  
    if scatter:
        plot_samples(gmm=gmm, ax=ax, N=N)
        
    set_ticksize(ax)
    set_titles(ax, xlabel=xlabel, ylabel=ylabel)


def calculate_summary_and_significance(estimates):
    """Given an array of estimates, calculate their sample mean, the error on the mean,
    and the significance with respect to 0.
    
    Parameters
    ----------
    estimates : 1D array-like
        The value of all estimates for a certain number of trials. 
        
    Returns
    -------
    mean_value : float
        The mean of the estimates.
    mean_error : float
        The error on the mean of the estimates.
    significance : float
        The distance of the mean with respect to 0.
    """
    trials_number = len(estimates)
    mean_value = np.mean(estimates)
    mean_error = np.sqrt(np.var(estimates, ddof=1))/np.sqrt(trials_number)
    # significance is with respect to 0
    significance = np.mean(estimates) / mean_error
    return mean_value, mean_error, significance


def histogram_estimates(estimates, ax=None, bins=20, alpha=1, title='',
                   legendsize=25, color='salmon', lw=3, histtype='step'):
    """Plot the histogram of an array of estimates, centred on 0.
    
    Parameters
    ----------
    estimates : 1D array-like
        The estimates whose histogram we plot. 
    ax : instance of the axes.Axes class from pyplot, default=None
        The panel where to draw the histogram. 
    bins : int, default=20
        Number of histogram bins
    alpha : float, default=1
        Transparency of the histogram line. Must be between 0 and 1.
    title : string, default=''
        Title of the panel.
    legendsize : int, default=25
         Font size of the legend.
    color : string, default='salmon'
        Histogram color.
    lw : int, default=3
        Line width of the histogram line.  
    histtype : {'bar', 'barstacked', 'step', 'stepfilled'}, default='step'
        The type of histogram to draw.
        
    Returns
    -------
    None
    """    
    ax = choose_ax(ax)
    mean_value, mean_error, significance = calculate_summary_and_significance(estimates)
    ax.hist(estimates.flatten(), alpha=alpha, bins=bins, color=color, histtype=histtype, lw=lw,
            label=f'MI values, {significance:.1f}$\sigma$,\n{mean_value:.4f}$\pm${mean_error:.4f}' 
            )
    ax.axvline(0, ls='--', lw=lw, c='black')
    set_ticksize(ax)
    set_titles(ax, xlabel='MI [nats]', ylabel='Counts', title=title)
    set_legend(ax, fontsize=legendsize)


def plot_loss_curve(loss_curves, index, ax):
    """Plot individual panel (referenced by an index) of the loss curves.
    
    Parameters
    ----------
    loss_curves : list of lists
        All loss curves, as returned by GMM_MI. Every pair of elements represents the train 
        and validation loss, respectively, for a particular fold and initialization.
    index : int
        The panel index where to draw the loss curves.
    ax : instance of the axes.Axes class from pyplot
        The current panel where to draw the loss curves. 
        
    Returns
    -------
    None
    """    
    current_train_loss = loss_curves[2*index]
    current_val_loss = loss_curves[2*index+1]
    iterations = np.arange(len(current_train_loss))

    ax.plot(iterations, current_train_loss, label='Train', lw=3, ls='-', color='grey')
    ax.plot(iterations, current_val_loss, label='Valid', lw=3, ls='-', color='salmon')
    
def plot_loss_curves(loss_curves, n_inits, n_folds, figsize=(20, 30), 
                     legendsize=18, frameon=False, loc='lower right'):
    """Plot all loss curves for the k-fold cross-validation in GMM_MI. The legend is placed only
    in the first panel. Every row corresponds to a different initialisation, and every column
    corresponds to a different fold.
    
    Parameters
    ----------
    loss_curves : list of lists
        All loss curves, as returned by GMM_MI. Every pair of elements represents the train 
        and validation loss, respectively, for a particular fold and initialization.
    n_inits : int
        The number of initialisations in the k-fold cross-validation.
    n_folds : int
        The number of folds in the k-fold cross-validation (k = n_folds).
    figsize : tuple of two integers, default=(20,30)
        The width and height of the entire plot, in inches.
    legendsize : int, default=18
        Font size of the legend (drawn only in the first panel).
    frameon : bool, default=False
        Whether to frame the legend or not.    
    loc : string, default='lower right'
        Position of the legend.     
        
    Returns
    -------
    None
    """
    fig, axes = plt.subplots(n_inits, n_folds, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i in range(n_inits*n_folds):
        ax = axes[i]
        plot_loss_curve(loss_curves, i, ax)
        if i == 0:
            set_legend(ax, fontsize=legendsize, frameon=frameon, loc=loc)
        if i >= (n_inits-1)*n_folds:
            set_titles(ax, xlabel='Iteration', fontsize=20)
        if i % n_folds == 0:
            set_titles(ax, ylabel='logL', fontsize=20)
        set_ticksize(ax)        
    fig.subplots_adjust(wspace=0.0, hspace=0.0)