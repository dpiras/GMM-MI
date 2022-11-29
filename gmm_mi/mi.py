import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
from gmm_mi.cross_validation import CrossValidation
from gmm_mi.single_fit import single_fit
from gmm_mi.param_holders import GMMFitParamHolder, SelectComponentsParamHolder, MIDistParamHolder


class EstimateMI:
    """Class to calculate mutual information (MI) distribution on 2D data, using Gaussian mixture models (GMMs).
    The main method is `fit` for continuous data, and `fit_categorical` for continuous-categorical data.
    The constructor parameters are mostly inherited from three classes: GMMFitParamHolder, and MIDistParamHolder

    Parameters
    ----------
    (inherited from GMMFitParamHolder, passed as gmm_fit_params)
    init_type : {'random', 'minmax', 'kmeans', 'randomized_kmeans', 'random_sklearn', 'kmeans_sklearn'}, default='random_sklearn'
        The method used to initialize the weights, the means, the covariances and the precisions in each CV fit.
        See utils.initializations for more details.
    scale : float, default=None
        The scale used for 'random', 'minmax' and 'randomized_kmeans' initializations. Not used in all other cases.
        See utils.initializations for more details.
    threshold_fit : float, default=1e-5
        The log-likelihood threshold on each GMM fit used to choose when to stop training.
        Smaller values will improve the fit quality and reduce the chances of stopping at a local optimum,
        while making the code considerably slower. This is equivalent to `tol` in sklearn GMMs.      
    reg_covar : float, default=1e-15
        The constant term added to the diagonal of the covariance matrices to avoid singularities.
        Smaller values will increase the chances of singular matrices, but will have a smaller impact
        on the final MI estimates. 
    max_iter : int, default=10000
        The maximum number of iterations in each GMM fit. We aim to stop only based on `threshold_fit`, 
        so it is set to a high value. A warning is raised if this threshold is reached; in that case, 
        simply increase this value, or check that you really need such a small `threshold_fit` value.
        
    (inherited from ChooseComponentParamHolder, passed as select_component_params)    
    n_inits : int, default=3
        Number of initializations used to find the best initialization parameters.
        Higher values will decrease the chances of stopping at a local optimum, while making the code slower.
    n_folds : int, default=2
        Number of folds in the cross-validation (CV) performed to find the best initialization parameters.
        A good value should ensure each fold has enough samples to be representative of your training set.
    metric_method : {'valid', 'aic', 'bic'}, default='valid'
        Metric used to select the optimal number of GMM components to perform density estimation.
        Must be one of:
            'valid': stop adding components when the validation log-likelihood stops increasing.
            'aic': stop adding components when the Akaike information criterion (AIC) stops decreasing.
            'bic': same as 'aic' but with the Bayesian information criterion (BIC).
    threshold_components : float, default=1e-5
        The metric threshold to decide when to stop adding GMM components. In other words, GMM-MI stops 
        adding components either when the metric gets worse, or when the improvement in the metric value
        is less than this threshold.
        Smaller values ensure that enough components are considered and thus that the data distribution is 
        correctly captured, while taking longer to converge and possibly insignificantly changing the 
        final value of MI.
    patience : int, default=1
        Number of extra components to "wait" until convergence is declared. Must be at least 1.
        Same concept as patience when training a neural network. Higher value will fit models
        with higher numbers of GMM components, while taking longer to converge.        
    max_components : int, default=50
        Maximum number of GMM components that is going to be tested. Hopefully stop much earlier than this.
        A warning is raised if this number of components is used; if so, you might want to use a different
        `metric_method`.
        
    (inherited from MIDistParamHolder, passed as mi_dist_params)            
    n_bootstrap : int, default=50 
        Number of bootstrap realisations to consider to obtain the MI uncertainty.
        Higher values will return a better estimate of the MI uncertainty, and
        will make the MI distribution more Gaussian-like, but the code will take longer.
        If < 1, do not perform bootstrap and actually just do a single fit on the entire dataset.
    MI_method : {'MC', 'quad'}, default='MC' 
        Method to calculate the MI integral. Must be one of:
            'MC': use Monte Carlo integration with MC_samples samples.
            'quad': use quadrature integration, as implemented in scipy, with default parameters.
    MC_samples : int, default=1e5
        Number of MC samples to use to estimate the MI integral. Only used if MI_method == 'MC'.
        Higher values will return less noisy estimates of MI, but will take longer.
    fixed_components_number : int, default=0
        The number of GMM components to use. If 0 (default), will do cross-validation to select 
        the number of components, and ignore this. Else, use this specified number of components,
        and ignore cross-validation.
        
    Attributes
    ----------
    converged : bool
        Flag to check whether we found the optimal number of components. Initially False.
    best_metric : float
        Metric value that is tracked to decide convergence with respect to the number of components.
        This can be either the validation log-likelihood, the AIC or BIC.
    patience_counter : int
        Counter to keep track of patience. Initially 0.
    results_dict : dict
        To keep track of the cross-validation results, and decide convergence.
    fixed_components : bool
        Set to True only if the user fixes a number of GMM components > 0.
    """
    def __init__(self, gmm_fit_params=None, select_components_params=None, mi_dist_params=None): 
        self.gmm_fit_params = gmm_fit_params if gmm_fit_params else GMMFitParamHolder()
        self.sel_comp_params = select_components_params if select_components_params else SelectComponentsParamHolder()
        self.mi_dist_params = mi_dist_params if mi_dist_params else MIDistParamHolder()
       
        self.converged = False
        self.best_metric = -np.inf
        self.patience_counter = 0
        self.results_dict = {}    
    
    def __getattr__(self, attr):
        """Access all hyperparameter classes methods and attributes."""
        try:
            return getattr(self.gmm_fit_params, attr)
        except:
            pass
        try:
            return getattr(self.sel_comp_params, attr)
        except:
            pass
        try:
            return getattr(self.mi_dist_params, attr)
        except:
            pass
    
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
        if self.metric_method == 'aic' or self.metric_method == 'bic':
            # in this case we need to re-fit the dataset; we start from the final point
            _, _, w_init, m_init, p_init = self._extract_best_parameters(n_components=n_components, 
                                                                   fixed_components=False, 
                                                                   patience=0)
            gmm = single_fit(X=self.X, n_components=n_components, reg_covar=self.reg_covar, 
                             threshold_fit=self.threshold_fit, max_iter=self.max_iter, 
                             w_init=w_init, m_init=m_init, p_init=p_init)  
            # this is an extra fit we make, so we also check if this converged
            convergence_flag = False if gmm.n_iter_ <= 2 else True
            self.results_dict[n_components]['convergence_flags'].append(convergence_flag)            
            if self.metric_method == 'aic':
                # negative since we maximise the metric
                metric = -gmm.aic(self.X)
            elif self.metric_method == 'bic':
                metric = -gmm.bic(self.X)
        elif self.metric_method == 'valid':
            metric = self.results_dict[n_components]['best_val_score']
        else:
            raise ValueError(f"metric_method must be either 'valid', 'aic' or 'bic, found '{self.metric_method}'")
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
            Best seed of the initialization that led to the highest validation log-likelihood.
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
        Convergence is declared when the metric has not improved, or when the improvement is
        less than the specified `threshold_components`.

        Parameters
        ----------
        n_components : int
            Number of GMM components being fitted. 
        """    
        if self.metric - self.best_metric > self.threshold_components:
            self.best_metric = self.metric
            if self.verbose:
                print(f'Current number of GMM components: {n_components}. Current metric: {self.best_metric:.3f}. Adding one component...')
        else:
            self.patience_counter += 1
            if self.verbose:
                print(f'Metric change is less than threshold; patience counter increased by 1...')
            if self.patience_counter >= self.patience:
                if self.verbose:
                    print(f'Reached patience limit, stop adding components.')
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
        """Perform bootstrap on the given data to calculate the distribution of mutual information (MI)
        in the continuous-continuous case. If n_bootstrap < 1, do only a single fit on the entire dataset.

        Parameters
        ----------
        n_components : int
            Number of GMM components to fit.
        random_state : int, default=None
            Random seed used to initialize the GMM model. 
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
            Mean of the MI distribution, in nat.
        MI_std : float
            Standard deviation of the MI distribution, in nat.
        """ 
        if self.n_bootstrap < 1:
            do_bootstrap = False
            self.n_bootstrap = 1 # we will use only one dataset, i.e. all data
        else:
            do_bootstrap = True
        MI_estimates = np.zeros(self.n_bootstrap)
        for n_b in tqdm(range(self.n_bootstrap)):
            if do_bootstrap:
                # we use index n_b to change the seed so that results will be fully reproducible
                rng = np.random.default_rng(n_b)
                X_bs = rng.choice(self.X, self.X.shape[0])
            else:
                X_bs = self.X
            gmm = single_fit(X=X_bs, n_components=n_components, reg_covar=self.reg_covar, 
                             threshold_fit=self.threshold_fit, max_iter=self.max_iter, 
                             random_state=random_state, w_init=w_init, m_init=m_init, p_init=p_init)
            current_MI_estimate = self._calculate_MI(gmm)
            MI_estimates[n_b] = current_MI_estimate
        MI_mean = np.mean(MI_estimates)
        MI_std = np.sqrt(np.var(MI_estimates, ddof=1)) if do_bootstrap else None
        return MI_mean, MI_std

    def fit(self, X, return_lcurves=False, verbose=False):
        """Calculate mutual information (MI) distribution (in nat).
        The first part performs density estimation of the data using GMMs and k-fold cross-validation.
        The second part uses the fitted model to calculate MI, using either Monte Carlo or quadrature methods.
        The MI uncertainty is calculated through bootstrap.

        Parameters
        ----------  
        X : array-like of shape (n_samples, 2)
            Samples from the joint distribution of the two variables whose MI is calculated.
        return_lcurves : bool, default=False
            Whether to return the loss curves or not (for debugging purposes).                          
        verbose : bool, default=False
            Whether to print useful procedural statements.
                
        Returns
        ----------
        MI_mean : float
            Mean of the MI distribution, in nat.
        MI_std : float
            Standard deviation of the MI distribution, in nat.
        loss_curves : list of lists
            Loss curves of the models trained during cross-validation; currently used for debugging.
            Only returned if return_lcurves is true.
        """
        assert X.shape[1] == 2, f"The shape of the data must be (n_samples, 2), found {X.shape}"    
        self.X = X
        self.verbose = verbose
        if self.verbose:
            if not self.fixed_components:
                print('Starting cross-validation procedure to select the number of GMM components...')
            else:
                print(f'Using {self.fixed_components_number} GMM components, as specified in input.')
        for n_components in range(1, self.max_components+1):
            if self.fixed_components:
                if n_components < self.fixed_components_number:
                    continue  
                else:
                    self.converged = True
            current_results_dict = CrossValidation(n_components=n_components, n_folds=self.n_folds, 
                                                   max_iter=self.max_iter, init_type=self.init_type, scale=self.scale,
                                                   n_inits=self.n_inits, threshold_fit=self.threshold_fit, 
                                                   reg_covar=self.reg_covar).fit(self.X)
            self.results_dict[n_components] = current_results_dict
            if not self.converged:
                self.metric = self._select_best_metric(n_components=n_components)
                # store metric, just to make it accessible outside of this class, if needed
                self.results_dict[n_components]['metric'] = self.metric
                self._check_convergence(n_components=n_components)

            if self.converged:
                best_components, best_seed, w_init, m_init, p_init = self._extract_best_parameters(n_components=n_components,    
                                                                                                  fixed_components=self.fixed_components,
                                                                                                  patience=self.patience)
                # these are assigned to self only to possibly plot the final model
                # in `plot_fitted_model`.
                self.best_components = best_components
                self.best_seed = best_seed
                self.w_init = w_init
                self.m_init = m_init
                self.p_init = p_init
                
                if self.verbose:
                    print(f'Convergence reached at {best_components} GMM components.') 
                    print(f'Starting MI integral estimation...') 

                # check if fits actually went on for a good amount of iterations
                if self.fixed_components:
                    convergence_flags = self.results_dict[self.best_components]['convergence_flags'] 
                else:
                    convergence_flags = [self.results_dict[n_c]['convergence_flags'] 
                                         for n_c in range(1, self.best_components+1)]
                # checking if all elements are False; in this case, a warning should be raised
                if not np.sum(convergence_flags): 
                    warnings.warn(    
                        f"The best-fit GMM parameters were found after only 2 iterations of the expectation-maximization procedure,"
                        " irrespective of the initial GMM parameters and number of GMM components. "
                        "This is usually suspicious, and might be a symptom of a bad fit. "
                        "Plot the loss curves as described in the walkthrough, and try reducing threshold_fit, "
                        "or with a different init_type.",
                        ConvergenceWarning,
                    )  
                    
                # get MI distribution
                MI_mean, MI_std = self._perform_bootstrap(n_components=best_components, random_state=best_seed, 
                                                             w_init=w_init, m_init=m_init, p_init=p_init)
                break

            # in the unlikely case that we have not reached enough GMM components,
            # we raise a ConvergenceWarning 
            if n_components == self.max_components:
                self.reached_max_components = True
                warnings.warn(f"Convergence in the number of GMM components was not reached. "\
                              f"Try increasing max_components or threshold_components, "\
                              f"or decreasing the patience.", ConvergenceWarning)
                best_components, best_seed, w_init, m_init, p_init = self._extract_best_parameters(n_components=self.max_components,    
                                                                                                  fixed_components=True,
                                                                                                  patience=0)
                # these are assigned to self only to possibly plot the final model
                # in `plot_fitted_model`.
                self.best_components = best_components
                self.best_seed = best_seed
                self.w_init = w_init
                self.m_init = m_init
                self.p_init = p_init
                
                # get MI distribution               
                MI_mean, MI_std = self._perform_bootstrap(n_components=best_components, random_state=best_seed, 
                                             w_init=w_init, m_init=m_init, p_init=p_init)                
        
        if self.verbose:
            print('MI estimation completed, returning mean and standard deviation.')
            
        if return_lcurves:
            return MI_mean, MI_std, self.lcurves        
        else:
            return MI_mean, MI_std    
 
    def plot_fitted_model(self, ax=None, **kwargs):
        """Fit model to inout data and plot its contours.
        Only works if the model has been fitted successfully first.
        
        Parameters
        ----------
        ax : instance of the axes.Axes class from pyplot, default=None
            The panel where to plot the samples. 
        kwargs : dictionary
            The extra keyword arguments to pass to the plotting function.
        
        Returns
        -------
        fig: instance of the figure.Figure class from pyplot
            The output figure.
        ax : instance of the axes.Axes class from pyplot
            The output panel.
        """
        assert (self.converged == True or self.reached_max_components == True), \
                                       "You can only plot the fitted model after MI has "\
                                       "been estimated; call .fit() on your data first!"
        from gmm_mi.utils.plotting import plot_gmm_contours
        gmm = single_fit(X=self.X, n_components=self.best_components, reg_covar=self.reg_covar, 
                 threshold_fit=self.threshold_fit, random_state=self.best_seed, max_iter=self.max_iter, 
                 w_init=self.w_init, m_init=self.m_init, p_init=self.p_init)
        ax = plot_gmm_contours(gmm, ax=ax, label='Fitted model', **kwargs)
        return ax
               
    def _calculate_MI_categorical(self):
        """Calculate mutual information (MI) integral given a Gaussian mixture model in 2D.
        Use only Monte Carlo (MC) method. 
        The complete formula can be found in Appendix B of Piras et al. (2022).

        Returns
        -------
        MI : float
            The value of MI.
        """    
        MI = 0 
        for category_value in range(self.category_values):
            samples = self.all_gmms[category_value].sample(self.MC_samples)[0]
            log_p = self.all_gmms[category_value].score_samples(samples)
            p_inner = 0
            for inner_category_value in range(self.category_values):
                p_inner += np.exp(self.all_gmms[inner_category_value].score_samples(samples))
            p = np.log(p_inner/self.category_values)
            MI += np.mean(log_p - p)
        MI /= self.category_values
        return MI
     
    def _perform_bootstrap_categorical(self):
        """Perform bootstrap on the given data to calculate the distribution of mutual information (MI),
        in the categorical-continuous case. If n_bootstrap < 1, do only a single fit on the entire dataset.

        Returns
        ----------
        MI_mean : float
            Mean of the MI distribution, in nat.
        MI_std : float
            Standard deviation of the MI distribution, in nat. None if bootstrap==False.
        """  
        if self.n_bootstrap < 1:
            do_bootstrap = False
            self.n_bootstrap = 1 # we will use only one dataset, i.e. all data
        else:
            do_bootstrap = True
        MI_estimates = np.zeros(self.n_bootstrap)
        for n_b in tqdm(range(self.n_bootstrap)):            
            # to store the fitted GMM models for each category value
            self.all_gmms = []
            for category_value in range(self.category_values):
                current_ids = np.where(self.y == category_value)
                # we select the relevant latents again
                current_latents = self.X[current_ids]
                current_latents = np.reshape(current_latents, (-1, 1))
                n_components = self.category_best_params[category_value]['bc']
                w_init = self.category_best_params[category_value]['w']
                m_init = self.category_best_params[category_value]['m']
                p_init = self.category_best_params[category_value]['p']
                random_state = self.category_best_params[category_value]['seed']
                if do_bootstrap:
                    # we use index n_b to change the seed so that results will be fully reproducible
                    rng = np.random.default_rng(n_b)
                    X_bs = rng.choice(current_latents, current_latents.shape[0])
                else:
                    X_bs = current_latents 
                gmm = single_fit(X=X_bs, n_components=n_components, reg_covar=self.reg_covar, 
                                 threshold_fit=self.threshold_fit, max_iter=self.max_iter, random_state=random_state, 
                                 w_init=w_init, m_init=m_init, p_init=p_init)               
                self.all_gmms.append(gmm)
            MI_estimates[n_b] = self._calculate_MI_categorical()

        MI_mean = np.mean(MI_estimates)
        MI_std = np.sqrt(np.var(MI_estimates, ddof=1)) if do_bootstrap else None
        return MI_mean, MI_std
       
    def fit_categorical(self, X, y, verbose=False):
        """Calculate mutual information (MI) distribution between a continuous and a categorical variable.
        The first part performs density estimation of the conditional distributions, using GMMs and k-fold cross-validation.
        The second part uses the fitted models to calculate MI, using Monte Carlo integration (numerical integration is not implemented).
        The MI uncertainty is calculated through bootstrap. Loss curves are not returned.
        The complete formula can be found in Appendix B of Piras et al. (2022).

        Parameters
        ----------  
        X : array-like of shape (n_samples)
            Samples of the continuous variable.
        y : array-like of shape (n_samples)   
            Categorical values corresponding to X.
        verbose : bool, default=False
            Whether to print useful procedural statements.
            
        Returns
        ----------
        MI_mean : float
            Mean of the MI distribution, in nat.
        MI_std : float
            Standard deviation of the MI distribution, in nat.
        """
        assert len(X.shape) == 1, f"The shape of the continuous data must be (n_samples), found {X.shape}"    
        assert len(y.shape) == 1, f"The shape of the categorical data must be (n_samples), found {y.shape}"
        assert X.shape[0] == y.shape[0], f"The length of the categorical and continuous data must be the same!"
        self.X = X           
        self.y = y 
        self.category_values = len(np.unique(y))
        self.verbose = verbose

        self.category_best_params = [] # used to store parameters of each GMM fit
        for category_value in range(self.category_values):
            current_ids = np.where(y == category_value)
            # select latents corresponding to current category value
            current_latents = X[current_ids]
            # need to fit the current latents; this is p(z_i | f_i = category_value)
            current_latents = np.reshape(current_latents, (-1, 1))
            
            # initialize attributes every time
            self.converged = False
            self.best_metric = -np.inf
            self.patience_counter = 0
            self.results_dict = {}  
        
            for n_components in range(1, self.max_components+1):
                if self.fixed_components:
                    if n_components < self.fixed_components_number:
                        continue  
                    else:
                        self.converged = True
                current_results_dict = CrossValidation(n_components=n_components, n_folds=self.n_folds, 
                                                       max_iter=self.max_iter, init_type=self.init_type, scale=self.scale,
                                                       n_inits=self.n_inits, threshold_fit=self.threshold_fit, 
                                                       reg_covar=self.reg_covar).fit(current_latents)
                self.results_dict[n_components] = current_results_dict
                if not self.converged:
                    self.metric = self._select_best_metric(n_components=n_components)
                    self._check_convergence(n_components=n_components)

                if self.converged:
                    best_components, best_seed, w_init, m_init, p_init = self._extract_best_parameters(n_components=n_components,                                                                                                       fixed_components=self.fixed_components,
                                                                                            patience=self.patience)
                    # save the best parameters for the current GMM category, and collect all of them before proceeding
                    self.category_best_params.append({'w': w_init, 'm': m_init, 'p': p_init, 
                                                 'bc': best_components, 'seed': best_seed})                    
                    break
            
            # in the unlikely case that we have not reached enough GMM components
            # we raise an error since we need all GMM parameters to calculate MI
            if n_components == self.max_components:
                raise ValueError(f"Convergence in the number of GMM components was not reached! "\
                                 f"Try increasing max_components or threshold_components, "\
                                 f"or decreasing the patience.")

        if self.verbose:
            print(f'Found best parameters for all GMMs, now onto the MI estimation') 
        
        # get MI distribution
        MI_mean, MI_std = self._perform_bootstrap_categorical()
        
        return MI_mean, MI_std 
        
