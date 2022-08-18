import numpy as np
from gmm_mi.cross_validation import CrossValidation
from gmm_mi.single_fit import single_fit


class EstimateMI:
    """Class to calculate mutual information (MI) distribution on 2D data, using Gaussian mixture models (GMMs).
    The main method is `fit`.

    Parameters
    ----------
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
        
    Attributes
    ----------
    converged : bool
        Flag to check whether we found the optimal number of components. Initially False.
    best_metric : float
        Metric that is tracked to decide convergence with respect to the number of components.
    patience_counter : int
        Counter to keep track of patience. Initially 0.
    results_dict : dict
        To keep track of the cross-validation results, and decide convergence.
    """
    def __init__(self, n_folds=2, n_inits=3, init_type='random_sklearn', reg_covar=1e-15, 
           tol=1e-5, max_iter=10000, max_components=100, select_c='valid', 
           patience=1, bootstrap=True, n_bootstrap=50, fixed_components=False, 
           fixed_components_number=1, MI_method='MC', MC_samples=1e5, return_lcurves=False, 
           verbose=False): 
        self.n_folds = n_folds
        self.n_inits = n_inits
        self.init_type = init_type
        self.reg_covar = reg_covar
        self.tol = tol
        self.max_iter = max_iter
        self.max_components = max_components
        self.select_c = select_c
        self.patience = patience
        self.bootstrap = bootstrap
        self.n_bootstrap = n_bootstrap
        self.fixed_components = fixed_components
        self.fixed_components_number = fixed_components_number
        self.MI_method = MI_method
        self.MC_samples = MC_samples
        self.return_lcurves = return_lcurves
        self.verbose = verbose

        self.converged = False
        self.best_metric = -np.inf
        self.patience_counter = 0
        self.results_dict = {}    
        assert select_c == 'valid' or select_c == 'aic' or select_c == 'bic', f"select_c must be either 'valid', 'aic' or 'bic, found '{select_c}'"
        
    def _select_best_metric(self, n_components):
        """Select best metric to choose the number of GMM components.

        Parameters
        ----------
        n_components : int
            Number of GMM components currently being fitted. 

        Returns
        ----------
        metric : float
            The value of the metric selected to choose the number of components.
        """    
        if self.select_c == 'aic' or self.select_c == 'bic':
            # in this case we need to re-fit the dataset; we start from the final point
            _, _, w_init, m_init, p_init = self._extract_best_parameters(n_components=n_components, 
                                                                   fixed_components=False, 
                                                                   patience=0)
            gmm = single_fit(X=self.X, n_components=n_components, reg_covar=self.reg_covar, 
                             tol=self.tol, max_iter=self.max_iter, 
                             w_init=w_init, m_init=m_init, p_init=p_init)                
            if self.select_c == 'aic':
                # negative since we maximise the metric
                metric = -gmm.aic(self.X)
            elif self.select_c == 'bic':
                metric = -gmm.bic(self.X)
        elif self.select_c == 'valid':
            metric = self.results_dict[n_components]['best_val_score']
        else:
            raise ValueError(f"select_c must be either 'valid', 'aic' or 'bic, found '{select_c}'")
        return metric        

    def _extract_best_parameters(self, n_components, fixed_components, patience):
        """Extract best parameters relative to cross-validation.

        Parameters
        ----------
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
        p_init : array-like of shape (n_components, 2, 2)
            Best GMM precisions (based on cross-validation), to be used as initial point of further fits.
        """    
        best_components = n_components
        if not fixed_components:
            best_components -= patience
        self.lcurves = self.results_dict[best_components]['all_lcurves']
        best_seed = self.results_dict[best_components]['best_seed']
        best_fold_in_init = self.results_dict[best_components]['best_fold_in_init']            
        all_ws = self.results_dict[best_components]['all_ws']
        all_ms = self.results_dict[best_components]['all_ms']
        all_ps = self.results_dict[best_components]['all_ps']
        w_init = all_ws[best_seed, best_fold_in_init]
        m_init = all_ms[best_seed, best_fold_in_init]
        p_init = all_ps[best_seed, best_fold_in_init] 
        return best_components, best_seed, w_init, m_init, p_init

    def _check_convergence(self, n_components):
        """Check if convergence with respect to the number of components has been reached.

        Parameters
        ----------
        n_components : int
            Number of GMM components being fitted. 
        """    
        if self.metric > self.best_metric:
            self.best_metric = self.metric
            if self.verbose:
                print(f'Current components: {n_components}. Current metric: {self.best_metric:.3f}. Adding one component...')
        else:
            self.patience_counter += 1
            if self.verbose:
                print(f'Metric did not improve; increasing patience counter...')
            if self.patience_counter >= self.patience:
                self.converged = True

    def _calculate_MI(self, gmm, tol_int=1.49e-8, limit=np.inf):
        """Calculate mutual information (MI) integral given a Gaussian mixture model in 2D.
        Use either Monte Carlo (MC) method, or quadrature method.

        Parameters
        ----------
        gmm : instance of GMM class
            The GMM model whose MI we calculate.
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
        if self.MI_method == 'MC':
            MI = gmm.estimate_MI_MC(MC_samples=self.MC_samples)
        elif MI_method == 'quad':
            MI = gmm.estimate_MI_quad(tol_int=tol_int, limit=limit)
        return MI

    def _perform_bootstrap(self, n_components, random_state, w_init, m_init, p_init):
        """Perform bootstrap on the given data to calculate the distribution of mutual information (MI).

        Parameters
        ----------
        n_components : int
            Number of GMM components to fit.
        random_state : int, default=None
            Random seed used to initialise the GMM model. 
            If initial GMM parameters are provided, used only to fix the trained model samples across trials.
        w_init : array-like of shape (n_components)
            Initial GMM weights.
        m_init : array-like of shape (n_components, 2)
            Initial GMM means.
        p_init : array-like of shape (n_components, 2, 2)
            Initial GMM precisions.

        Returns
        ----------
        MI_mean : float
            Mean of the MI distribution.
        MI_std : float
            Standard deviation of the MI distribution.
        """  
        MI_estimates = np.zeros(self.n_bootstrap)
        for n_b in range(self.n_bootstrap):
            # we use index n_b to change the seed so that results will be fully reproducible
            rng = np.random.default_rng(n_b)
            X_bs = rng.choice(self.X, self.X.shape[0])
            gmm = single_fit(X=X_bs, n_components=n_components, reg_covar=self.reg_covar, 
                             tol=self.tol, max_iter=self.max_iter, random_state=random_state, 
                             w_init=w_init, m_init=m_init, p_init=p_init)
            current_MI_estimate = self._calculate_MI(gmm)
            MI_estimates[n_b] = current_MI_estimate
        MI_mean = np.mean(MI_estimates)
        MI_std = np.sqrt(np.var(MI_estimates, ddof=1))
        return MI_mean, MI_std

    def fit(self, X):
        """Calculate mutual information (MI) distribution.
        The first part performs density estimation of the data using GMMs and k-fold cross-validation.
        The second part uses the fitted model to calculate MI, using either Monte Carlo or quadrature methods.
        The MI uncertainty is calculated through bootstrap.

        Parameters
        ----------  
        X : array-like of shape (n_samples, 2)
            Samples from the joint distribution of the two variables whose MI is calculated.
                          
        Returns
        ----------
        MI_mean : float
            Mean of the MI distribution.
        MI_std : float
            Standard deviation of the MI distribution.
        loss_curves : list of lists
            Loss curves of the models trained during cross-validation; currently used for debugging.
            Only returned if return_lcurves is true.
        """
        assert X.shape[1] == 2, f"The shape of the data must be (n_samples, 2), found {X.shape}"    
        self.X = X
        for n_components in range(1, self.max_components+1):
            if self.fixed_components:
                if n_components < self.fixed_components_number:
                    continue  
                else:
                    self.converged = True
            current_results_dict = CrossValidation(n_components=n_components, n_folds=self.n_folds, 
                                                   max_iter=self.max_iter, init_type=self.init_type, 
                                                   n_inits=self.n_inits, tol=self.tol, 
                                                   reg_covar=self.reg_covar).fit(self.X)
            self.results_dict[n_components] = current_results_dict
            if not self.converged:
                self.metric = self._select_best_metric(n_components=n_components)
                self._check_convergence(n_components=n_components)

            if self.converged:
                best_components, best_seed, w_init, m_init, p_init = self._extract_best_parameters(n_components=n_components,    
                                                                                                  fixed_components=self.fixed_components,
                                                                                                  patience=self.patience)
                if self.verbose:
                    print(f'Convergence reached at {best_components} components') 
                if self.bootstrap:
                    MI_mean, MI_std = self._perform_bootstrap(n_components=best_components, random_state=best_seed, 
                                                             w_init=w_init, m_init=m_init, p_init=p_init)
                else:
                    # only now while debugging, we perform a single fit on the entire dataset
                    gmm = single_fit(X=self.X, n_components=best_components, reg_covar=self.reg_covar, 
                                     tol=self.tol, random_state=best_seed, max_iter=self.max_iter, 
                                     w_init=w_init, m_init=m_init, p_init=p_init)
                    MI_mean = self._calculate_MI(gmm)
                    MI_std = None
                break

        if self.return_lcurves:
            return MI_mean, MI_std, self.lcurves        
        else:
            return MI_mean, MI_std    
       