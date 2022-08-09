import numpy as np
from gmm import GMM
from utils.initializations import initialize_parameters
from sklearn.model_selection import KFold
import warnings


def _run_cross_validation(X, kf, val_scores, all_ws, all_ms, all_ps, all_lcurves, n_components, 
                          n_folds=3, n_inits=5, init_type='random_sklearn', 
                          reg_covar=1e-15, tol=1e-6, max_iter=10000):        
    """
    Actually run the cross-validation (CV) procedure, filling all arrays and lists with results.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, 2)
        Data on which CV is performed.
    kf : KFold class instance
        The k-fold cross-validator object, as obtained from sklearn.    
    val_scores : array-like of shape (n_inits, n_folds)
        Contains all validation log-likelihood values ("scores"). Will be filled in this function.
    all_ws : array-like of shape (n_inits, n_folds, n_components)
        Contains all final GMM weights. Will be filled in this function.
    all_ms : array-like of shape (n_inits, n_folds, n_components, 2)
        Contains all final GMM means. Will be filled in this function.    
    all_ps : array-like of shape (n_inits, n_folds, n_components, 2, 2)
        Contains all final GMM precision matrices. Will be filled in this function.   
    all_lcurves : list of lists
        Contains all loss curves. Will be filled in this function.
    n_components : int
        Number of GMM components currently being fitted.
    n_folds : int, default=3
        Number of folds.
    n_inits : int, default=5
        Number of initializations.
    init_type : {'random', 'minmax', 'kmeans', 'random_sklearn', 'kmeans_sklearn'}, default='random_sklearn'
        The method used to initialize the weights, the means, the covariances and the precisions in each fit.
        See utils.initializations for more details.
    reg_covar : float, default=1e-15
        The constant term added to the diagonal of the covariance matrices to avoid singularities.
    tol : float, default=1e-6
        The log-likelihood threshold on each GMM fit used to choose when to stop training.
    max_iter : int, default=10000
        The maximum number of iterations in each GMM fit. We aim to stop only based on the tolerance, 
        so it is set to a high value.    

    Returns
    ----------
    val_scores : array-like of shape (n_inits, n_folds)
        Filled array with all validation log-likelihood values ("scores").
    all_ws : array-like of shape (n_inits, n_folds, n_components)
        Filled array with all final GMM weights
    all_ms : array-like of shape (n_inits, n_folds, n_components, 2)
        Filled array with all final GMM means.    
    all_ps : array-like of shape (n_inits, n_folds, n_components, 2, 2)
        Filled array with all final GMM precision matrices. 
    all_lcurves : list of lists
        Filled list with all loss curves.   
    """
    for random_state in range(n_inits):
        # initialise with different seed random_state
        w_init, m_init, c_init, p_init = initialize_parameters(X, random_state=random_state, n_components=n_components, 
                                                               init_type=init_type)       
        # perform k-fold CV
        for k_id, (train_indices, valid_indices) in enumerate(kf.split(X)):
            X_training = X[train_indices]
            X_validation = X[valid_indices]            
            gmm = single_fit(X=X_training, n_components=n_components, reg_covar=reg_covar, tol=tol, 
                       max_iter=max_iter, random_state=random_state, w_init=w_init, m_init=m_init, 
                       p_init=p_init, val_set=X_validation)
            # we take the mean logL per sample, since folds might have slightly different sizes
            val_score = gmm.score_samples(X_validation).mean()            
            # save current scores, as well as parameters
            val_scores[random_state, k_id] = np.copy(val_score)
            all_ws[random_state, k_id] = np.copy(gmm.weights_)
            all_ms[random_state, k_id] = np.copy(gmm.means_)
            all_ps[random_state, k_id] = np.copy(gmm.precisions_)
            # save the loss functions as well
            all_lcurves.append(np.copy(gmm.train_loss))
            all_lcurves.append(np.copy(gmm.val_loss))
 
    return val_scores, all_ws, all_ms, all_ps, all_lcurves
    
    
def cross_validation(X, n_components, n_folds=3, n_inits=5, init_type='random_sklearn', 
                     reg_covar=1e-15, tol=1e-6, max_iter=10000):
    """
    Perform cross-validation (CV) to select the best GMM initialization parameters, 
    and thus avoid local minima in the density estimation.

    Parameters
    ----------
    X : array-like of shape (n_samples, 2)
        Data on which CV is performed.
    n_components : int
        Number of GMM components currently being fitted.
    n_folds : int, default=3
        Number of folds.
    n_inits : int, default=5
        Number of initializations.
    init_type : {'random', 'minmax', 'kmeans', 'random_sklearn', 'kmeans_sklearn'}, default='random_sklearn'
        The method used to initialize the weights, the means, the covariances and the precisions in each fit.
        See utils.initializations for more details.
    reg_covar : float, default=1e-15
        The constant term added to the diagonal of the covariance matrices to avoid singularities.
    tol : float, default=1e-6
        The log-likelihood threshold on each GMM fit used to choose when to stop training.
    max_iter : int, default=10000
        The maximum number of iterations in each GMM fit. We aim to stop only based on the tolerance, 
        so it is set to a high value.    

    Returns
    ----------
    results_dict : dict
        A dictionary containing all results from the CV procedure:
        'best_seed': the initialization seed that led to the highest mean log-likelihood across folds.
        'best_val_score': the corresponding value of the highest mean log-likelihood.
        'best_fold_in_init':  the index of the best fold within the best initialization.
        'all_lcurves': list of list with the training and validation curves.
        'all_ws': all weights of the fitted GMM models.
        'all_ms': all means of the fitted GMM models.
        'all_ps': all precisions of the fitted GMM models.
    """
    # create empty arrays to store validation log-likelihoods
    val_scores = np.zeros((n_inits, n_folds))    
    # create empty arrays for final GMM parameters, and loss curves
    all_ws = np.zeros((n_inits, n_folds, n_components))
    all_ms = np.zeros((n_inits, n_folds, n_components, 2))
    all_ps = np.zeros((n_inits, n_folds, n_components, 2, 2))    
    all_lcurves = []
    # random split seed is fixed here, but results should be independent of the exact split
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)    
    val_scores, all_ws, all_ms, all_ps, all_lcurves = _run_cross_validation(X=X, kf=kf, val_scores=val_scores, 
                                                                           all_ws=all_ws, all_ms=all_ms, all_ps=all_ps,
                                                                           all_lcurves=all_lcurves, 
                                                                           n_components=n_components, n_folds=n_folds, 
                                                                           n_inits=n_inits, max_iter=max_iter,
                                                                           init_type=init_type, reg_covar=reg_covar, tol=tol)      
    # select seed with highest val score across the different inits
    avg_val_scores = np.mean(val_scores, axis=1)
    best_seed = np.argmax(avg_val_scores)
    best_val_score = np.max(avg_val_scores)
    # within the best fold, also select the model with the highest validation logL
    best_fold_in_init = np.argmax(val_scores[best_seed])    
    results_dict = {'best_seed': best_seed, 'best_val_score': best_val_score, 
                    'best_fold_in_init': best_fold_in_init, 'all_lcurves': all_lcurves, 
                    'all_ws': all_ws, 'all_ms': all_ms, 'all_ps': all_ps}
    return results_dict        
    

def single_fit(X, n_components, reg_covar=1e-15, tol=1e-6, max_iter=10000, 
                random_state=None, w_init=None, m_init=None, p_init=None, val_set=None):
    """
    Perform a single fit of a GMM on input data.

    Parameters
    ----------
    X : array-like of shape (n_samples, 2)
        Data to fit.
    n_components : int
        Number of GMM components to fit.
    reg_covar : float, default=1e-15
        The constant term added to the diagonal of the covariance matrices to avoid singularities.
    tol : float, default=1e-6
        The log-likelihood threshold on each GMM fit used to choose when to stop training.
    max_iter : int, default=10000
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
    val_set : array-like of shape (n_samples, 2), default=None
        Validation set. If provided, also loss curves are considered.

    Returns
    ----------
    gmm : instance of GMM class
        The fitted GMM model.
    """
    gmm = GMM(n_components=n_components, reg_covar=reg_covar, 
                tol=tol, max_iter=max_iter, 
                random_state=random_state, weights_init=w_init, 
                means_init=m_init, precisions_init=p_init).fit(X, val_set=val_set)
    return gmm
 
    
def select_best_metric(X, results_dict, n_components,
                       select_c='valid', init_type='random_sklearn',
                       reg_covar=1e-15, tol=1e-6, max_iter=10000):
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
    tol : float, default=1e-6
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

              
    
def GMM_MI(X, n_folds=3, n_inits=5, init_type='random_sklearn', reg_covar=1e-15, 
           tol=1e-6, max_iter=10000, max_components=100, select_c='valid', 
           patience=1, bootstrap=True, n_bootstrap=100, fixed_components=False, 
           fixed_components_number=1, MI_method='MC', MC_samples=1e5, verbose=False): 
    """
    Calculate mutual information (MI) distribution on 2D data, using Gaussian mixture models (GMMs).
    The first part performs density estimation of the data using GMMs and k-fold cross-validation.
    The second part uses the fitted model to calculate MI, using either Monte Carlo or quadrature methods.
    The MI uncertainty is calculated through bootstrap.

    Parameters
    ----------
    X : array-like of shape (n_samples, 2)
        Samples from the joint distribution of the two variables whose MI is calculated.
    n_folds : int, default=3
        Number of folds in the cross-validation (CV) performed to find the best initialization parameters.
    n_inits : int, default=5
        Number of initializations used to find the best initialization parameters.
    init_type : {'random', 'minmax', 'kmeans', 'random_sklearn', 'kmeans_sklearn'}, default='random_sklearn'
        The method used to initialize the weights, the means, the covariances and the precisions in each CV fit.
        See utils.initializations for more details.
    reg_covar : float, default=1e-15
        The constant term added to the diagonal of the covariance matrices to avoid singularities.
    tol : float, default=1e-6
        The log-likelihood threshold on each GMM fit used to choose when to stop training.
    max_iter : int, default=10000
        The maximum number of iterations in each GMM fit. We aim to stop only based on the tolerance, 
        so it is set to a high value.    
    max_components : int, default=100
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
    n_bootstrap : int, default=100 
        Number of bootstrap realisations to consider to obtain the MI uncertainty.
    
    ed_components : bool, default=False 
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
    assert select_c == 'valid' or select_c == 'aic' or select_c == 'bic', f"select_c must be either 'valid', 'aic' or 'bic, found '{select_c}'"
    for n_components in range(1, max_components+1):
        if fixed_components:
            if n_components < fixed_components_number:
                continue  
            else:
                converged = True
        current_results_dict = cross_validation(X=X, n_components=n_components, n_folds=n_folds, max_iter=max_iter,
                                       init_type=init_type, n_inits=n_inits, tol=tol, reg_covar=reg_covar)
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
            
    return MI_mean, MI_std, lcurves