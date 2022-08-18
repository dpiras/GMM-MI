import numpy as np
from gmm_mi.cross_validation import CrossValidation
from gmm_mi.single_fit import single_fit

    
def select_best_metric(X, results_dict, n_components,
                       select_c='valid', init_type='random_sklearn',
                       reg_covar=1e-15, tol=1e-5, max_iter=10000):
    """
    Select best metric to choose the number of GMM components.

    Parameters
    ----------
    X : array-like of shape (n_samples, 2)
        Samples from the joint distribution of the two variables whose MI is calculated.
    results_dict : dictionary
        Contains all the results from cross-validation.
    n_components : int
        Number of GMM components to fit.           
    select_c : {'valid', 'aic', 'bic'}, default='valid'
        Method used to select the optimal number of GMM components to perform density estimation.
        Must be one of:
            'valid': stop adding components when the validation log-likelihood stops increasing.
            'aic': stop adding components when the Akaike information criterion (AIC) stops decreasing
            'bic': same as 'aic' but with the Bayesian information criterion (BIC).    
    init_type : {'random', 'minmax', 'kmeans', 'random_sklearn', 'kmeans_sklearn'}, default='random_sklearn'
        The method used to initialize the weights, the means, the covariances and the precisions in each CV fit.
        See utils.initializations for more details.
    reg_covar : float, default=1e-15
        The constant term added to the diagonal of the covariance matrices to avoid singularities.
    tol : float, default=1e-5
        The log-likelihood threshold on each GMM fit used to choose when to stop training.
    max_iter : int, default=10000
        The maximum number of iterations in each GMM fit. We aim to stop only based on the tolerance, 
        so it is set to a high value.      
        
    Returns
    ----------
    metric : float
        The value of the metric selected to choose the number of components.
    """    
    if select_c == 'aic' or select_c == 'bic':
        # in this case we need to re-fit the dataset; we start from the final point
        _, _, w_init, m_init, p_init, _ = extract_best_parameters(n_components=n_components, 
                                                               fixed_components=False, 
                                                               patience=0, 
                                                               results_dict=results_dict)
        gmm = single_fit(X=X, n_components=n_components, reg_covar=reg_covar, 
                         tol=tol, max_iter=max_iter, w_init=w_init, 
                         m_init=m_init, p_init=p_init)                
        if select_c == 'aic':
            # negative since we maximise the metric
            metric = -gmm.aic(X)
        elif select_c == 'bic':
            metric = -gmm.bic(X)
    elif select_c == 'valid':
        metric = results_dict[n_components]['best_val_score']
    else:
        raise ValueError(f"select_c must be either 'valid', 'aic' or 'bic, found '{select_c}'")
    return metric

def check_convergence(metric, best_metric, n_components, patience, patience_counter, verbose=False):
    """
    Check if convergence with respect to the number of components has been reached.

    Parameters
    ----------
    metric : float
        Current metric value, to choose convergence.
    best_metric : float
        Current best metric value.
    n_components : int
        Number of GMM components being fitted. 
    patience : int
        Number of extra components to "wait" until convergence is declared.
    patience_counter : int
        Counter to keep track of how many components we added so far with respect to the patience.
    verbose : bool, default=False
        Whether to print useful procedural statements.
        
    Returns
    ----------
    converged : bool
        Whether we reached the optimal number of GMM components.
    best_metric : float
        Updated best metric.
    patience_counter : int
        Updated patience counter.
    """    
    converged = False
    if metric > best_metric:
        best_metric = metric
        if verbose:
            print(f'Current components: {n_components}. Current metric: {best_metric:.3f}. Adding one component...')
    else:
        patience_counter += 1
        if verbose:
            print(f'Metric did not improve; increasing patience counter...')
        if patience_counter >= patience:
            converged = True
    return converged, best_metric, patience_counter

def extract_best_parameters(results_dict, n_components, fixed_components, patience):
    """
    Extract best parameters relative to cross-validation.
    These parameters are then used to initialize the final MI estimation.

    Parameters
    ----------
    results_dict : dictionary
        Contains all the results from cross-validation.
    n_components : int
        Number of GMM components to fit. 
    fixed_components : bool
        Fix the number of GMM components to use for density estimation.
        In this instance, only used to decide if patience is used or not.
    patience : int
        Number of extra components to "wait" until convergence is declared. 
        Decides how far back to look to select the number of components. Only used if fixed_components is not True.
    
    Returns
    ----------
    best_components : int
        Best number of components to fit (either the input ones, or fewer if it's during patience).
    best_seed : int
        Best seed of the initialisation that led to the highest validation log-likelihood.
    w_init : array-like of shape (n_components)
        Best GMM weights (based on cross-validation), to be used as initial point of further fits.
    m_init : array-like of shape (n_components, 2)
        Best GMM means (based on cross-validation), to be used as initial point of further fits.
    p_init : array-like of shape ( n_components, 2, 2)
        Best GMM precisions (based on cross-validation), to be used as initial point of further fits.
    lcurves : list of lists
        Filled list with all loss curves from cross-validation. 
    """    
    best_components = n_components
    if not fixed_components:
        best_components -= patience
    lcurves = results_dict[best_components]['all_lcurves']
    best_seed = results_dict[best_components]['best_seed']
    best_fold_in_init = results_dict[best_components]['best_fold_in_init']            
    all_ws = results_dict[best_components]['all_ws']
    all_ms = results_dict[best_components]['all_ms']
    all_ps = results_dict[best_components]['all_ps']
    w_init = all_ws[best_seed, best_fold_in_init]
    m_init = all_ms[best_seed, best_fold_in_init]
    p_init = all_ps[best_seed, best_fold_in_init] 
    return best_components, best_seed, w_init, m_init, p_init, lcurves


def calculate_MI(gmm, MI_method='MC', MC_samples=1e5, tol_int=1.49e-8, limit=np.inf):
    """
    Calculate mutual information (MI) integral given a Gaussian mixture model in 2D.
    Use either Monte Carlo (MC) method, or quadrature method.

    Parameters
    ----------
    gmm : instance of GMM class
        The GMM model whose MI we calculate.
    MI_method : {'MC', 'quad'}, default='MC'
        Whether to use MC ('MC') method or quadrature ('quad') method to estimate the integral.
    MC_samples : int, default=1e5
        Number of MC samples used to estimate integral. Only used when MI_method='MC'.
    tol_int : float, default=1.49e-8
        Integral tolerance; the default value is the one form scipy. 
        Only used when MI_method='quad'.
    limit : float, default=np.inf
        The extrema of the integral to calculate. Usually the whole plane, so defaults to inf.
        Only used when MI_method='quad'. Integral goes from -limit to +limit.

    Returns
    ----------
    MI : float
        The value of MI.
    """    
    if MI_method == 'MC':
        MI = gmm.estimate_MI_MC(MC_samples=MC_samples)
    elif MI_method == 'quad':
        MI = gmm.estimate_MI_quad(tol_int=tol_int, limit=limit)
    return MI


def perform_bootstrap(X, n_bootstrap, n_components, 
                      reg_covar, tol, max_iter, random_state,
                      w_init, m_init, p_init, MI_method='MC', MC_samples=1e5):
    """
    Perform bootstrap on the given data to calculate the distribution of mutual information (MI).

    Parameters
    ----------
    X : array-like of shape (n_samples, 2)
        Data over which bootstrap is performed.
    n_bootstrap : int
        Number of bootstrap realisations to consider to obtain the MI uncertainty.
    n_components : int
        Number of GMM components to fit.
    reg_covar : float
        The constant term added to the diagonal of the covariance matrices to avoid singularities.
    tol : float
        The log-likelihood threshold on each GMM fit used to choose when to stop training.
    max_iter : int
        The maximum number of iterations in each GMM fit. We aim to stop only based on the tolerance, 
        so it is set to a high value.    
    random_state : int, default=None
        Random seed used to initialise the GMM model. 
        If initial GMM parameters are provided, used only to fix the trained model samples across trials.
    w_init : array-like of shape (n_components)
        Initial GMM weights.
    m_init : array-like of shape (n_components, 2)
        Initial GMM means.
    p_init : array-like of shape ( n_components, 2, 2)
        Initial GMM precisions.
    MI_method : {'MC', 'quad'}, default='MC'
        Whether to use MC ('MC') method or quadrature ('quad') method to estimate the integral.
    MC_samples : int, default=1e5
        Number of MC samples used to estimate integral. Only used when MI_method='MC'.

    Returns
    ----------
    MI_mean : float
        Mean of the MI distribution.
    MI_std : float
        Standard deviation of the MI distribution.
    """  
    MI_estimates = np.zeros(n_bootstrap)
    for n_b in range(n_bootstrap):
        # we use index n_b to change the seed so that results will be fully reproducible
        rng = np.random.default_rng(n_b)
        X_bs = rng.choice(X, X.shape[0])
        gmm = single_fit(X=X_bs, n_components=n_components, reg_covar=reg_covar, 
                    tol=tol, max_iter=max_iter, 
                    random_state=random_state, w_init=w_init, 
                    m_init=m_init, p_init=p_init)
        current_MI_estimate = calculate_MI(gmm, MI_method=MI_method, MC_samples=MC_samples)
        MI_estimates[n_b] = current_MI_estimate
    MI_mean = np.mean(MI_estimates)
    MI_std = np.sqrt(np.var(MI_estimates, ddof=1))
    return MI_mean, MI_std

              
def GMM_MI(X, n_folds=2, n_inits=3, init_type='random_sklearn', reg_covar=1e-15, 
           tol=1e-5, max_iter=10000, max_components=100, select_c='valid', 
           patience=1, bootstrap=True, n_bootstrap=50, fixed_components=False, 
           fixed_components_number=1, MI_method='MC', MC_samples=1e5, return_lcurves=False, 
           verbose=False): 
    """
    Calculate mutual information (MI) distribution on 2D data, using Gaussian mixture models (GMMs).
    The first part performs density estimation of the data using GMMs and k-fold cross-validation.
    The second part uses the fitted model to calculate MI, using either Monte Carlo or quadrature methods.
    The MI uncertainty is calculated through bootstrap.

    Parameters
    ----------
    X : array-like of shape (n_samples, 2)
        Samples from the joint distribution of the two variables whose MI is calculated.
    n_folds : int, default=2
        Number of folds in the cross-validation (CV) performed to find the best initialization parameters.
    n_inits : int, default=3
        Number of initializations used to find the best initialization parameters.
    init_type : {'random', 'minmax', 'kmeans', 'random_sklearn', 'kmeans_sklearn'}, default='random_sklearn'
        The method used to initialize the weights, the means, the covariances and the precisions in each CV fit.
        See utils.initializations for more details.
    reg_covar : float, default=1e-15
        The constant term added to the diagonal of the covariance matrices to avoid singularities.
    tol : float, default=1e-5
        The log-likelihood threshold on each GMM fit used to choose when to stop training.
    max_iter : int, default=10000
        The maximum number of iterations in each GMM fit. We aim to stop only based on the tolerance, 
        so it is set to a high value.    
    max_components : int, default=50
        Maximum number of GMM components that is going to be tested. Hopefully stop much earlier than this.   
    select_c : {'valid', 'aic', 'bic'}, default='valid'
        Method used to select the optimal number of GMM components to perform density estimation.
        Must be one of:
            'valid': stop adding components when the validation log-likelihood stops increasing.
            'aic': stop adding components when the Akaike information criterion (AIC) stops decreasing
            'bic': same as 'aic' but with the Bayesian information criterion (BIC).
    patience : int, default=1, 
        Number of extra components to "wait" until convergence is declared. 
        Same concept as patience when training a neural network.
    bootstrap : bool, default=True, 
        Whether to perform bootstrap or not to get the uncertainty on MI. 
        If False, only a single value of MI is returned.
    n_bootstrap : int, default=50 
        Number of bootstrap realisations to consider to obtain the MI uncertainty.   
    fixed_components : bool, default=False 
        Fix the number of GMM components to use for density estimation.
        For debugging purposes, or when you are sure of how many components to use.
    fixed_components_number : int, default=1
        The number of GMM components to use. Only used if fixed_components == True.
    MI_method : {'MC', 'quad'}, default='MC' 
        Method to calculate the MI integral. Must be one of:
            'MC': use Monte Carlo integration with MC_samples samples.
            'quad': use quadrature integration, as implemented in scipy, with default parameters.
    MC_samples : int, default=1e5
        Number of MC samples to use to estimate the MI integral. Only used if MI_method == 'MC'.
    return_lcurves : bool, default=False
        Whether to return the loss curves or not (for debugging purposes).
    verbose : bool, default=False
        Whether to print useful procedural statements.
        
    Returns
    ----------
    MI_mean : float
        Mean of the MI distribution.
    MI_std : float
        Standard deviation of the MI distribution.
    loss_curves : list of lists
        Loss curves of the models trained during cross-validation; currently used for debugging.
    """
    converged = False
    best_metric = -np.inf
    patience_counter = 0
    results_dict = {}    
    assert X.shape[1] == 2, f"The shape of the data must be (n_samples, 2), found {X.shape}"
    assert select_c == 'valid' or select_c == 'aic' or select_c == 'bic', f"select_c must be either 'valid', 'aic' or 'bic, found '{select_c}'"
    for n_components in range(1, max_components+1):
        if fixed_components:
            if n_components < fixed_components_number:
                continue  
            else:
                converged = True
        current_results_dict = CrossValidation(n_components=n_components, n_folds=n_folds, max_iter=max_iter,
                                       init_type=init_type, n_inits=n_inits, tol=tol, reg_covar=reg_covar).fit(X)
        results_dict[n_components] = current_results_dict
        if not converged:
            metric = select_best_metric(X=X, results_dict=results_dict, select_c=select_c,
                                        n_components=n_components, init_type=init_type, tol=tol, max_iter=max_iter)
            converged, best_metric, patience_counter = check_convergence(metric=metric, best_metric=best_metric, 
                                                                         n_components=n_components, 
                                                                         patience=patience,
                                                                         patience_counter=patience_counter, 
                                                                         verbose=verbose)
               
        if converged:
            best_components, best_seed, w_init, m_init, p_init, lcurves = extract_best_parameters(results_dict=results_dict,
                                                                                                  n_components=n_components,    
                                                                                                  fixed_components=fixed_components,
                                                                                                  patience=patience)
            if verbose:
                print(f'Convergence reached at {best_components} components') 
            if bootstrap:
                MI_mean, MI_std = perform_bootstrap(X=X, n_bootstrap=n_bootstrap, n_components=best_components, 
                                                    reg_covar=reg_covar, tol=tol, max_iter=max_iter, 
                                                    random_state=best_seed, MI_method=MI_method, 
                                                    MC_samples=MC_samples, w_init=w_init, m_init=m_init, p_init=p_init)
            else:
                # only now while debugging, we perform a single fit on the entire dataset
                gmm = single_fit(X=X, n_components=best_components, reg_covar=reg_covar, 
                                 tol=tol, random_state=best_seed, max_iter=max_iter, 
                                 w_init=w_init, m_init=m_init, p_init=p_init)
                MI_mean = calculate_MI(gmm, MI_method=MI_method, MC_samples=MC_samples)
                MI_std = None
            break
    
    if return_lcurves:
        return MI_mean, MI_std, lcurves        
    else:
        return MI_mean, MI_std
    