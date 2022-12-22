import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from tqdm import tqdm
from gmm_mi.cross_validation import CrossValidation
from gmm_mi.single_fit import single_fit
from gmm_mi.param_holders import GMMFitParamHolder, SelectComponentsParamHolder, MIDistParamHolder

class EstimateMI:
    """Class to calculate mutual information (MI) distribution on 2D data, using Gaussian mixture models (GMMs).
    It can also calculate the KL divergence between the two marginal variables.
    The main method is `fit` for continuous data, and `fit_categorical` for continuous-categorical data.
    Then `estimate` can be used to compute MI, KL or both. To directly obtain MI estimate, use `fit_estimate`.
    The constructor parameters are mostly inherited from two classes: GMMFitParamHolder and ChooseComponentParamHolder. 
    The class MIDistParamHolder is also used to pass arguments to the `estimate`, so that it is possible to repeat the MI
    estimation without having to re-fit the data.

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
    fixed_components_number : int, default=0
        The number of GMM components to use. If 0 (default), will do cross-validation to select 
        the number of components, and ignore this. Else, use this specified number of components,
        and ignore cross-validation.
        
    Attributes
    ----------
    fit_done : bool
        Check `fit` method has been called. Initially False. Has to be true to call `estimate`.
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
    def __init__(self, gmm_fit_params=None, select_components_params=None): 
        self.gmm_fit_params = gmm_fit_params if gmm_fit_params else GMMFitParamHolder()
        self.sel_comp_params = select_components_params if select_components_params else SelectComponentsParamHolder()
       
        self.fit_done = False
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
    
    def _check_shapes(self, X, Y):
        """ Check that the shapes of the arrays given as input to GMM-MI are either 2D or 1D,
        and return the correct array to give as input.

        Parameters
        ----------  
        X : array-like of shape (n_samples, 2), (n_samples, 1) or (n_samples)
            Samples from the joint distribution of the two variables whose MI or KL is calculated.
            If Y is None, must be of shape (n_samples, 2); otherwise, it must be either (n_samples, 1) or (n_samples).
        Y : array-like of shape (n_samples, 1) or (n_samples), default=None
            Samples from the marginal distribution of one of the two variables whose MI or KL is calculated.
            If None, X must be of shape (n_samples, 2); otherwise, X and Y must be (n_samples, 1) or (n_samples).
        
        Returns
        ----------
        X : array-like of shape (n_samples, 2)
            The 2D array that is used to estimate MI or KL, with the expected shape.     
        """
        if len(X.shape) == 1:
            X = np.reshape(X, (X.shape[0], 1)) # add extra dimension       
        if Y is None:
            if X.shape[1] != 2:
                raise ValueError(f"Y is None, but the input array X is not 2-dimensional. "\
                     f"In this case, both X and Y should be 1-dimensional.")
            else:
                return X
        # if Y is not None, we can manipulate it    
        else:
            if len(Y.shape) == 1:
                Y = np.reshape(Y, (Y.shape[0], 1)) # add extra dimension
            if X.shape[1] == 1 and Y.shape[1] == 1:
                X = np.hstack((X, Y))
                return X
            else:
                raise ValueError(f"Y is not None, but the input arrays X or Y are not 1-dimensional. "\
                     f"Shapes found: {X.shape}, {Y.shape}.")
                
    def _select_best_metric(self, n_components):
        """Select best metric to choose the number of GMM components.
        Note all metrics are calculated and stored, but only the one indicated
        in input is used to make decisions on convergence.

        Parameters
        ----------
        n_components : int
            Number of GMM components currently being fitted. 

        Returns
        ----------
        metric : float
            The value of the metric selected to choose the number of components.
        """    
        # for aic and bic we need to re-fit the dataset; we start from the final point
        _, _, w_init, m_init, p_init = self._extract_best_parameters(n_components=n_components, 
                                                               fixed_components=False, 
                                                               patience=0)
        gmm = single_fit(X=self.X, n_components=n_components, reg_covar=self.reg_covar, 
                         threshold_fit=self.threshold_fit, max_iter=self.max_iter, 
                         w_init=w_init, m_init=m_init, p_init=p_init)  
        # this is an extra fit we make, so we also check if this converged
        convergence_flag = False if gmm.n_iter_ <= 2 else True
        self.results_dict[n_components]['convergence_flags'].append(convergence_flag)  
        aic = gmm.aic(self.X)
        self.results_dict[n_components]['aic'] = aic
        bic = gmm.bic(self.X)
        self.results_dict[n_components]['bic'] = bic
        valid_score = self.results_dict[n_components]['best_val_score']
        self.results_dict[n_components]['valid_score'] = valid_score
        if self.metric_method == 'aic':
            # negative since we maximise the metric
            metric = -aic
        elif self.metric_method == 'bic':
            # negative since we maximise the metric
            metric = -bic
        elif self.metric_method == 'valid':
            metric = valid_score
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
            Only used when integral_method='quad'.
        limit : float, default=np.inf
            The extrema of the integral to calculate. Usually the whole plane, so defaults to inf.
            Only used when integral_method='quad'. Integral goes from -limit to +limit.

        Returns
        ----------
        MI : float
            The value of MI, in the units specified by the base provided as input to the `estimate` method.
        """    
        if self.integral_method == 'MC':
            MI = gmm.estimate_MI_MC(MC_samples=self.MC_samples)
        elif self.integral_method == 'quad':
            MI = gmm.estimate_MI_quad(tol_int=tol_int, limit=limit)
        return MI
    
    def _calculate_KL(self, gmm, kl_order='forward', tol_int=1.49e-8, limit=np.inf):
        """Calculate KL divergence given a Gaussian mixture model in 2D.
        Use either Monte Carlo (MC) method, or quadrature method.

        Parameters
        ----------
        gmm : instance of GMM class
            The GMM model between whose marginal the KL is calculated.
        kl_order : one of {'forward', 'reverse'}, default='forward'
            Whether to calculate the KL divergence between p(x) and p(y), or between p(y) and p(x).
        tol_int : float, default=1.49e-8
            Integral tolerance; the default value is the one form scipy.
            Only used when integral_method='quad'.
        limit : float, default=np.inf
            The extrema of the integral to calculate. Usually the whole plane, so defaults to inf.
            Only used when integral_method='quad'. Integral goes from -limit to +limit.
           
        Returns
        ----------
        KL : float
            The value of KL, in the units specified by the base provided as input to the `estimate` method.
        """
        if self.integral_method == 'MC':
            KL = gmm.estimate_KL_MC(kl_order=kl_order, MC_samples=self.MC_samples)
        elif self.integral_method == 'quad':
            KL = gmm.estimate_KL_quad(kl_order=kl_order, tol_int=tol_int, limit=limit)
        return KL

    def _perform_bootstrap(self, n_components, random_state, w_init, m_init, p_init, include_kl=False, kl_order='forward'):
        """Perform bootstrap on the given data to calculate the distribution of mutual information (MI) or KL
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
        include_kl : bool, default=False
            If True, compute KL divergence too. This is not returned, but can be accessed as an attribute.
        kl_order : one of {'forward', 'reverse'}, default='forward'
            Whether to calculate the KL divergence between p(x) and p(y), or between p(y) and p(x).

        Returns
        ----------
        MI_mean : float
            Mean of the MI distribution.
        MI_std : float
            Standard deviation of the MI distribution.
        """ 
        if self.n_bootstrap < 1:
            do_bootstrap = False
            self.n_bootstrap = 1 # we will use only one dataset, i.e. all data
        else:
            do_bootstrap = True
        
        if include_kl:
            KL_estimates = np.zeros(self.n_bootstrap)
            if self.verbose:
                print(f'Computing also {kl_order} KL divergence.')
        else:
            self.KL_mean, self.KL_std = None, None
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
            if include_kl:
                current_KL_estimate = self._calculate_KL(gmm, kl_order=kl_order) 
                KL_estimates[n_b] = current_KL_estimate
        MI_mean = np.mean(MI_estimates)
        MI_std = np.sqrt(np.var(MI_estimates, ddof=1)) if do_bootstrap else None
        if include_kl:
            self.KL_mean = np.mean(KL_estimates)
            self.KL_std = np.sqrt(np.var(KL_estimates, ddof=1)) if do_bootstrap else None            
        return MI_mean, MI_std
    
    def _set_units(self, MI_mean, MI_std, base):
        """ Set units according to input base.
        
        Parameters
        ----------
        MI_mean : float
            Mean of the MI distribution, in nat.
        MI_std : float
            Standard deviation of the MI distribution, in nat.
        base : float
            The base of the logarithm to calculate MI or KL. 
            
        Returns
        ----------
        MI_mean : float
            Mean of the MI distribution, in the units set by base.
        MI_std : float
            Standard deviation of the MI distribution, in the units set by base.
        """
        if MI_mean is not None:            
            MI_mean /= np.log(base)
        if MI_std is not None:
            MI_std /= np.log(base)
        return MI_mean, MI_std

    def fit(self, X, Y=None, verbose=False):
        """Performs density estimation of the data using GMMs and k-fold cross-validation.
        The fitted model will be used to estimate MI and/or KL.

        Parameters
        ----------  
        X : array-like of shape (n_samples, 2) or (n_samples, 1) or (n_samples)
            Samples from the joint distribution of the two variables whose MI or KL is calculated.
            If Y is None, must be of shape (n_samples, 2); otherwise, it must be either (n_samples, 1) or (n_samples).
        Y : array-like of shape (n_samples, 1) or (n_samples), default=None
            Samples from the marginal distribution of one of the two variables whose MI or KL is calculated.
            If None, X must be of shape (n_samples, 2); otherwise, X and Y must be (n_samples, 1) or (n_samples).                   
        verbose : bool, default=False
            Whether to print useful procedural statements.
                
        Returns
        ----------
        None
        """ 
        self.verbose = verbose
        # if fit was already done, exit without doing anything
        if self.fit_done:
            if self.verbose:
                print('Fit already done, not being repeated.')
            return
        
        # check shapes and proceed with fit
        X = self._check_shapes(X, Y)
        self.X = X
        
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
                # these are assigned to self only to possibly plot the final model in `plot_fitted_model`.
                self.best_components = best_components
                self.best_seed = best_seed
                self.w_init = w_init
                self.m_init = m_init
                self.p_init = p_init

                if self.verbose:
                    print(f'Convergence reached at {best_components} GMM components.')

                # check if fits actually went on for a good amount of iterations
                if self.fixed_components:
                    convergence_flags = self.results_dict[self.best_components]['convergence_flags']
                else:
                    convergence_flags = [self.results_dict[n_c]['convergence_flags']
                                         for n_c in range(1, self.best_components+1)]
                # checking if all elements are False; in this case, a warning should be raised
                if not np.sum(convergence_flags):
                    warnings.warn(
                        f"All CV GMM fits converged only after their second iteration for all components; "
                        "this is usually suspicious, and might be a symptom of a bad fit. "
                        "Plot the loss curves as described in the walkthrough, and try reducing threshold_fit, "
                        "or with a different init_type.",
                        ConvergenceWarning,
                    )

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
        
        # at this point, either self.converged=True or self.reached_max_components=True
        # these indicate that the fit has been done, and there is no need to repeat it
        self.fit_done = self.converged or self.reached_max_components
                
    def estimate(self, mi_dist_params=None, base=np.exp(1), include_kl=False, kl_order='forward', verbose=False):
        """Calculate mutual information (MI) distribution (in nat, unless a different base is specified).
        Uses the fitted model to estimate MI using either Monte Carlo or quadrature methods.
        It can also estimate the KL divergence between the marginals, in the preferred order (KL is not symmetric).
        Uncertainties on MI and KL are calculated through bootstrap.

        Parameters
        ----------                            
        (inherited from MIDistParamHolder, passed as mi_dist_params)            
        n_bootstrap : int, default=50 
            Number of bootstrap realisations to consider to obtain the MI uncertainty.
            Higher values will return a better estimate of the MI uncertainty, and
            will make the MI distribution more Gaussian-like, but the code will take longer.
            If < 1, do not perform bootstrap and actually just do a single fit on the entire dataset.
        integral_method : {'MC', 'quad'}, default='MC' 
            Method to calculate the MI or KL integral. Must be one of:
                'MC': use Monte Carlo integration with MC_samples samples.
                'quad': use quadrature integration, as implemented in scipy, with default parameters.
        MC_samples : int, default=1e5
            Number of MC samples to use to estimate the MI integral. Only used if integral_method == 'MC'.
            Higher values will return less noisy estimates of MI, but will take longer.        
        
        include_kl : bool, default=False
            If True, compute KL divergence too. This is not returned, but can be accessed as an attribute.
        kl_order : one of {'forward', 'reverse'}, default='forward'
            Whether to calculate the KL divergence between p(x) and p(y) ('forward')
            or between p(y) and p(x) ('reverse').
        base : float, default=np.exp(1)
            The base of the logarithm to calculate MI or KL. 
            By default, unit is nat. Set base=2 for bit.
        verbose : bool, default=False
            Whether to print useful procedural statements.
            
        Returns
        ----------
        MI_mean : float
            Mean of the MI distribution, in nat (unless a different base is specified).
        MI_std : float
            Standard deviation of the MI distribution, in nat (unless a different base is specified).
        """ 
        self.verbose = verbose
        
        if not self.fit_done:
            raise NotFittedError(
                "This EstimateMI instance is not fitted yet. Call `fit` with appropriate "
                "arguments before using this estimator."
                                )

        self.mi_dist_params = mi_dist_params if mi_dist_params else MIDistParamHolder()
            
        # get MI distribution
        MI_mean, MI_std = self._perform_bootstrap(n_components=self.best_components, random_state=self.best_seed,
                                                  w_init=self.w_init, m_init=self.m_init, p_init=self.p_init, 
                                                  include_kl=include_kl, kl_order=kl_order)
        
        if self.verbose:
            print('MI estimation completed, returning mean and standard deviation.')
        
        # set units according to input base
        MI_mean, MI_std = self._set_units(MI_mean, MI_std, base)
        self.MI_mean, self.MI_std = MI_mean, MI_std
        # also set the units of KL;
        self.KL_mean, self.KL_std = self._set_units(self.KL_mean, self.KL_std, base)
        return MI_mean, MI_std    
     
    def fit_estimate(self, X, Y=None, mi_dist_params=None, include_kl=False, kl_order='forward', base=np.exp(1), verbose=False):
        """Combine the `fit` and `estimate` methods for easier calculation of MI. 
        See the respective methods for all information.
        """ 
        self.fit(X=X, Y=Y, verbose=verbose)
        MI_mean, MI_std = self.estimate(mi_dist_params=mi_dist_params, include_kl=include_kl, 
                                       kl_order=kl_order, base=base, verbose=verbose)
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
        if not self.fit_done:
            raise NotFittedError(
                "This EstimateMI instance is not fitted yet. Call `fit` with appropriate "
                "arguments before using this estimator."
                                )
            
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
            Mean of the MI distribution.
        MI_std : float
            Standard deviation of the MI distribution. None if bootstrap==False.
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
                current_ids = np.where(self.Y == category_value)
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
       
    def fit_categorical(self, X, Y=None, verbose=False):
        """Fit joint probability distribution between a continuous and a categorical variable.
        Uses GMMs and k-fold cross-validation.

        Parameters
        ----------  
        X : array-like of shape (n_samples, 2) or (n_samples, 1) or (n_samples)
            Samples from the joint distribution of the two variables whose MI is calculated.
            If Y is None, must be of shape (n_samples, 2); otherwise, it must be either (n_samples, 1) or (n_samples).
            The first column of X must correspond to the samples of the continuous variable, and the second column
            to the categorical values corresponding to the first column.
        Y : array-like of shape (n_samples, 1) or (n_samples), default=None
            Categorical values corresponding to X, if X is 1D.
            If None, X must be of shape (n_samples, 2); otherwise, X and Y must be (n_samples, 1) or (n_samples).     
            
        Returns
        ----------
        None
        """
        self.verbose = verbose
        # if fit was already done, exit without doing anything
        if self.fit_done:
            if self.verbose:
                print('Fit already done, not being repeated.')
            return
        
        X_together = self._check_shapes(X, Y)
        X = X_together[:, :1]
        Y = X_together[:, 1:]       
        self.X = X
        self.Y = Y
        self.category_values = len(np.unique(Y))
        self.verbose = verbose

        self.category_best_params = [] # used to store parameters of each GMM fit
        for category_value in range(self.category_values):
            current_ids = np.where(self.Y == category_value)
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
                    best_components, best_seed, w_init, m_init, p_init = self._extract_best_parameters(n_components=n_components, 
                                                                                          fixed_components=self.fixed_components,
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

        # at this point, either self.converged=True or self.reached_max_components=True
        # these indicate that the fit has been done, and there is no need to repeat it
        self.fit_done = self.converged or self.reached_max_components
        
        if self.verbose:
            print(f'Found best parameters for all GMMs, now can start MI estimation.') 
        
    def estimate_categorical(self, mi_dist_params=None, base=np.exp(1), verbose=False):
        """Calculate mutual information (MI) distribution between a continuous and a categorical variable.
        Uses the fitted models to calculate MI, using Monte Carlo integration (numerical integration is not implemented).
        The MI uncertainty is calculated through bootstrap.
        The complete formula can be found in Appendix B of Piras et al. (2022), arXiv 2211.00024.

        Parameters
        ----------  
        (inherited from MIDistParamHolder, passed as mi_dist_params)            
        n_bootstrap : int, default=50 
            Number of bootstrap realisations to consider to obtain the MI uncertainty.
            Higher values will return a better estimate of the MI uncertainty, and
            will make the MI distribution more Gaussian-like, but the code will take longer.
            If < 1, do not perform bootstrap and actually just do a single fit on the entire dataset.
        integral_method : {'MC'}, default='MC' 
            Method to calculate the MI integral. Must be 'MC' (uses Monte Carlo integration with 
            MC_samples samples). Only 'MC' is implemented for the categorical case; any other choice 
            will throw an error. 
        MC_samples : int, default=1e5
            Number of MC samples to use to estimate the MI integral. Only used if integral_method == 'MC'.
            Higher values will return less noisy estimates of MI, but will take longer.        
        base : float, default=np.exp(1)
            The base of the logarithm to calculate MI. 
            By default, unit is nat. Set base=2 for bit.
        verbose : bool, default=False
            Whether to print useful procedural statements.

            
        Returns
        ----------
        MI_mean : float
            Mean of the MI distribution, in nat (unless a different base is specified).
        MI_std : float
            Standard deviation of the MI distribution, in nat (unless a different base is specified).
        """
        self.verbose = verbose
        
        if not self.fit_done:
            raise NotFittedError(
                "This EstimateMI instance is not fitted yet. Call `fit` with appropriate "
                "arguments before using this estimator."
                                )

        self.mi_dist_params = mi_dist_params if mi_dist_params else MIDistParamHolder()
        if self.integral_method != 'MC':
            raise ValueError(
                "Only MC integration is implemented for the categorical case. "
                f"Set integral_method='MC'; found {self.integral_method}"
                                )            

        # get MI distribution
        MI_mean, MI_std = self._perform_bootstrap_categorical()
        
        # set units according to input base
        MI_mean, MI_std = self._set_units(MI_mean, MI_std, base)
        self.MI_mean, self.MI_std = MI_mean, MI_std        
        return MI_mean, MI_std 
        
    def fit_estimate_categorical(self, X, Y, mi_dist_params=None, base=np.exp(1), verbose=False):
        """Combine the `fit_categorical` and `estimate_categorical` methods for easier calculation of MI. 
        See the respective methods for all information.
        """ 
        self.fit_categorical(X=X, Y=Y, verbose=verbose)
        MI_mean, MI_std = self.estimate_categorical(mi_dist_params=mi_dist_params, base=base, verbose=verbose)
        return MI_mean, MI_std  