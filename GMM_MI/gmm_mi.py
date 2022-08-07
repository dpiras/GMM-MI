import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from scipy import linalg
import scipy.integrate as integrate
from sklearn.mixture import GaussianMixture as GMM
from sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob, _compute_precision_cholesky, _estimate_gaussian_covariances_full
from sklearn.utils import check_random_state
from sklearn.model_selection import KFold
from sklearn.exceptions import ConvergenceWarning
from utils.initializations import initialize_parameters
import warnings


def log_pdf(y, x, model):
    """Calculate scale prefactor ('s') for the 'random' and 'minmax' initializations.
    
    Parameters
    ----------
    X : array-like of shape  (n_samples, n_features)
        The data based on which the scale is calculated. 
    n_components : int
        Number of GMM components.
        
    Returns
    ----------
    scale : float
        Scale prefactor, to be used to initialize the GMM.
    """
    y = np.array(y)
    x = np.array(x)
    X = np.concatenate((y.reshape(1, 1), x.reshape(1, 1))).T
    return model.score_samples(X)


def pdf(y, x):
    return np.exp(log_pdf(y, x))


def entropy_2d_integrand(y, x, model):
    """
    TODO
    """
    logp = log_pdf(y, x, model)
    p = np.exp(logp)
    return p*logp


def entropy_1d_integrand(x, model, index):
    """
    TODO
    """
    # add check that index is either 0 or 1
    x = np.array(x)
    w = model.weights_
    m = model.means_[:, index:index+1]
    c = model.covariances_[:, index:index+1, index:index+1]
    gmm_marginal = GMM(n_components=len(w), weights_init=w, means_init=m, covariances_init=c)
    logp_1d = gmm_marginal.score_samples(x.reshape(-1, 1))
    p = np.exp(logp_1d)
    return p*logp_1d


class GMM(GMM):
    """
    Custom GMM class based on the sklearn GMM class.
    This allows to work with a GMM with fixed parameters, without fitting it.
    It also allows to estimate MI with a certain number of MC samples, or with quadrature method.
    The different initialisation types are dealt with separately.
    """
    def __init__(self,
                 n_components=1,
                 covariance_type="full",
                 tol=1e-5,
                 reg_covar=1e-6,
                 max_iter=100,
                 n_init=1,
                 init_params="random",
                 random_state=None,
                 warm_start=False,
                 verbose=0,
                 verbose_interval=10,
                 weights_init=None,
                 means_init=None,
                 precisions_init=None,
                 covariances_init=None
                 ):
        super(GMM, self).__init__(n_components=n_components,
                 covariance_type=covariance_type,
                 tol=tol,
                 reg_covar=reg_covar,
                 max_iter=max_iter,
                 n_init=n_init,
                 init_params=init_params,
                 random_state=random_state,
                 warm_start=warm_start,
                 verbose=verbose,
                 verbose_interval=verbose_interval,
                 weights_init=weights_init,
                 means_init=means_init,
                 precisions_init=precisions_init,
                )

        self.means_ = means_init
        if covariances_init is None:
            covariances_init = np.linalg.inv(self.precisions_init)
        self.covariances_ = covariances_init
        self.covariances_init = covariances_init
        self.weights_ = weights_init
        self.covariance_type = covariance_type
        self.precisions_cholesky_ = _compute_precision_cholesky(
                self.covariances_, self.covariance_type
            )


    def score_samples(self, X):
        """Compute the log-likelihood of each sample.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log-likelihood of each sample in `X` under the current model.
        """

        return logsumexp(self._estimate_weighted_log_prob(X), axis=1)

    def predict(self, X):
        """Predict the labels for the data samples in X using trained model.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """

        return self._estimate_weighted_log_prob(X).argmax(axis=1)

    def predict_proba(self, X):
        """Evaluate the components' density for each sample.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        resp : array, shape (n_samples, n_components)
            Density of each Gaussian component for each sample in X.
        """
        _, log_resp = self._estimate_log_prob_resp(X)
        return np.exp(log_resp)

    def sample(self, n_samples=1):
        """Generate random samples from the fitted Gaussian distribution.
        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.
        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample.
        y : array, shape (nsamples,)
            Component labels.
        """

        if n_samples < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at "
                "least one sample." % (self.n_components)
            )

        _, n_features = self.means_.shape
        rng = check_random_state(self.random_state)
        n_samples_comp = rng.multinomial(n_samples, self.weights_)

        if self.covariance_type == "full":
            X = np.vstack(
                [
                    rng.multivariate_normal(mean, covariance, int(sample))
                    for (mean, covariance, sample) in zip(
                        self.means_, self.covariances_, n_samples_comp
                    )
                ]
            )
        elif self.covariance_type == "tied":
            X = np.vstack(
                [
                    rng.multivariate_normal(mean, self.covariances_, int(sample))
                    for (mean, sample) in zip(self.means_, n_samples_comp)
                ]
            )
        else:
            X = np.vstack(
                [
                    mean + rng.randn(sample, n_features) * np.sqrt(covariance)
                    for (mean, covariance, sample) in zip(
                        self.means_, self.covariances_, n_samples_comp
                    )
                ]
            )

        y = np.concatenate(
            [np.full(sample, j, dtype=int) for j, sample in enumerate(n_samples_comp)]
        )

        return (X, y)

    def score_samples_marginal(self, X, index=0):
        """Compute the log-likelihood of each sample for the marginal model, indexed by either 0 (x) or 1 (y).
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        index: integer
            Either 0 (marginal x) or 1 (marginal y).
        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log-likelihood of each sample in `X` under the current model.
        """

        oned_cholesky = np.sqrt(1/self.covariances_[:, index, index]).reshape(-1, 1, 1)
        marginal_logprob = _estimate_log_gaussian_prob(
            X, self.means_[:, index].reshape(-1, 1), oned_cholesky, self.covariance_type
        )

        return logsumexp(np.log(self.weights_) + marginal_logprob, axis=1)

    def fit(self, X, y=None, val_set=None):
        """TODO
        Estimate model parameters with the EM algorithm.
        The method fits the model ``n_init`` times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for ``max_iter``
        times until the change of likelihood or lower bound is less than
        ``tol``, otherwise, a ``ConvergenceWarning`` is raised.
        If ``warm_start`` is ``True``, then ``n_init`` is ignored and a single
        initialization is performed upon the first call. Upon consecutive
        calls, training starts where it left off.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
            The fitted mixture.
        """
        self.fit_predict(X, y, val_set)
        return self
    
    def fit_predict(self, X, y=None, val_set=None):
        """TODO
        Estimate model parameters using X and predict the labels for X.
        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
        raised. After fitting, it predicts the most probable label for the
        input data points.
        .. versionadded:: 0.20
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self._check_initial_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.inf
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, random_state)

            lower_bound = -np.inf if do_init else self.lower_bound_
            if val_set is not None:
                self.train_loss, self.val_loss = [], []
                
            for n_iter in range(1, self.max_iter + 1):

                prev_lower_bound = lower_bound

                log_prob_norm, log_resp = self._e_step(X)
                self._m_step(X, log_resp)
                lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)

                change = lower_bound - prev_lower_bound
                self._print_verbose_msg_iter_end(n_iter, change)

                if val_set is not None:
                    log_prob_norm_train, _ = self._e_step(X)
                    log_prob_norm_val, _ = self._e_step(val_set)
                    self.train_loss.append(log_prob_norm_train)
                    self.val_loss.append(log_prob_norm_val)

                if abs(change) < self.tol:
                    self.converged_ = True
                    break

            self._print_verbose_msg_init_end(lower_bound)

            if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                max_lower_bound = lower_bound
                best_params = self._get_parameters()
                best_n_iter = n_iter

        if not self.converged_:
            warnings.warn(
                "Initialization %d did not converge. "
                "Try different init parameters, "
                "or increase max_iter, tol "
                "or check for degenerate data." % (init + 1),
                ConvergenceWarning,
            )

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._e_step(X)

        return log_resp.argmax(axis=1)
        
    def estimate_MI_MC(self, MC_samples=1e3):
        """
        Compute the mutual information (MI) associated with a particular GMM model, using MC integration
        Parameters
        ----------
        MC_samples : integer
            Number of Monte Carlo samples to perform numerical integration of the MI integral.
        Returns
        ----------
        MI : integer
            The value of mutual information.
        -------
        """
        # sample MC samples
        points, clusters = self.sample(MC_samples)
        
        # we first evaluate the log-likelihood for the joint probability
        joint = self.score_samples(points)

        # we then evaluate the marginals; index=0 corresponds to x, index=y corresponds to y
        marginal_x = self.score_samples_marginal(points[:, :1], index=0)
        marginal_y = self.score_samples_marginal(points[:, 1:], index=1)

        MI = np.mean(joint - marginal_x - marginal_y)
        return MI

    def estimate_MI_quad(self, tol_int=1.49e-8, limit=np.inf):
        """
        TO DO:
        Compute the mutual information (MI) associated with a particular GMM model, using quadrature integration.
        Parameters
        ----------
        MC_samples : integer
            Number of Monte Carlo samples to perform numerical integration of the MI integral.
        Returns
        ----------
        MI : integer
            The value of mutual information.
        -------
        """
        gmm = GMM(n_components=self.n_components, weights_init=self.weights_, 
                     means_init=self.means_, covariances_init=self.covariances_)
        
        entropy_2d = integrate.dblquad(entropy_2d_integrand, -limit, limit, 
                                      lambda x: -limit, lambda x: limit, 
                                      args=[gmm], epsabs=tol_int, epsrel=tol_int)[0]
        entropy_1d_x = integrate.quad(entropy_1d_integrand, -limit, limit, 
                                      args=(gmm, 0), epsabs=tol_int, epsrel=tol_int)[0]
        entropy_1d_y = integrate.quad(entropy_1d_integrand, -limit, limit, 
                                      args=(gmm, 1), epsabs=tol_int, epsrel=tol_int)[0]

        MI = entropy_2d - entropy_1d_x - entropy_1d_y
        return MI


def single_cross_validation(X, kf, validation_scores, 
                            all_ws, all_ms, all_ps, all_loss_curves,
                            val_scores_seeds, n_components, n_folds, n_inits, 
                            max_iter, init_type, reg_covar, tol):    
    
    for r in range(n_inits):
        # initialise with different seed r
        w_init, m_init, c_init, p_init = initialize_parameters(X, random_state=r, 
                                                               n_components=n_components, init_type=init_type)
        
        # perform k-fold CV
        for k_idx, (train_indices, valid_indices) in enumerate(kf.split(X)):
            X_training = X[train_indices]
            X_validation = X[valid_indices]
            
            gmm = single_fit(X_training, n_components=n_components, reg_covar=reg_covar, tol=tol, 
                       max_iter=max_iter, random_state=r, w_init=w_init, m_init=m_init, 
                       p_init=p_init, val_set=X_validation)

            # we take the mean logL per sample, since folds might have slightly different sizes
            val_score = gmm.score_samples(X_validation).mean()
            
            # save current scores, as well as parameters
            validation_scores[r, k_idx] = np.copy(val_score)
            all_ws[r, k_idx] = np.copy(gmm.weights_)
            all_ms[r, k_idx] = np.copy(gmm.means_)
            all_ps[r, k_idx] = np.copy(gmm.precisions_)
            # save the loss functions as well
            all_loss_curves.append(np.copy(gmm.train_loss))
            all_loss_curves.append(np.copy(gmm.val_loss))

        # take mean of current seed's val scores
        val_scores_seeds[r] = np.mean(validation_scores[r])
    
    return val_scores_seeds, validation_scores, all_ws, all_ms, all_ps, all_loss_curves
    
    
def cross_validation(X, n_components=1, n_folds=5, n_inits=5, max_iter=10000, 
                 init_type='random_sklearn', reg_covar=1e-6, tol=1e-6):
    """
    Docstring TODO
    """
    # fix number of components to true model
    # create empty arrays for multiple initialisations
    val_scores_seeds = np.zeros(n_inits)    
    # these are one for each init and each fold; we'll average over these at the end of the CV
    validation_scores = np.zeros((n_inits, n_folds))    
    # prepare the folds; note the splitting will be the same for all initialisations
    # the random seed is fixed here, but results should be independent of the exact split
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    # also create emoty arrays for final GMM parameters
    all_ws = np.zeros((n_inits, n_folds, n_components))
    all_ms = np.zeros((n_inits, n_folds, n_components, 2))
    all_ps = np.zeros((n_inits, n_folds, n_components, 2, 2))    
    all_loss_curves = []
    val_scores_seeds, validation_scores, all_ws, all_ms, all_ps, all_loss_curves = single_cross_validation(X=X, kf=kf,                                                                                                          validation_scores=validation_scores, 
                                                                                     all_ws=all_ws, all_ms=all_ms, 
                                                                                     all_ps=all_ps, all_loss_curves=all_loss_curves,
                                                                                     val_scores_seeds=val_scores_seeds,
                                                                                     n_components=n_components, n_folds=n_folds, 
                                                                                     n_inits=n_inits, max_iter=max_iter,
                                                                                     init_type=init_type, reg_covar=reg_covar, tol=tol)  
    
    # select seed with highest val score across the different inits
    best_seed = np.argmax(val_scores_seeds)
    best_val_score = np.max(val_scores_seeds)
    # within the best fold, select the model with the highest validation logL
    best_fold_in_init = np.argmax(validation_scores[best_seed])    
    results_dict = {'best_seed': best_seed, 'best_fold_in_init': best_fold_in_init, 
                   'best_val_score': best_val_score, 'all_loss_curves': all_loss_curves, 
                   'all_ws': all_ws, 'all_ms': all_ms, 'all_ps': all_ps}
    return results_dict        
    

def single_fit(X, n_components, reg_covar, tol, max_iter, 
                random_state, w_init, m_init, p_init, val_set=None):
    gmm = GMM(n_components=n_components, reg_covar=reg_covar, 
                tol=tol, max_iter=max_iter, 
                random_state=random_state, weights_init=w_init, 
                means_init=m_init, precisions_init=p_init).fit(X, val_set=val_set)
    return gmm
    
def select_best_metric(X, current_results_dict, select_c, 
                       random_state, n_components, init_type,
                      tol, max_iter):
    # if we want to select the number of components based on AIC or BIC, we need to change the metric
    if select_c == 'aic' or select_c == 'bic':
        current_seed = current_results_dict['best_seed']
        w_init, m_init, c_init, p_init = initialize_parameters(X, random_state=current_seed, 
                                                               n_components=n_components, init_type=init_type)
        gmm = single_fit(X, n_components=n_components, reg_covar=reg_covar, 
                tol=tol, max_iter=max_iter, random_state=current_seed, w_init=w_init, 
                m_init=m_init, p_init=p_init)                
        if select_c == 'aic':
            # explain why we use the negative metric for AIC and BIC
            metric = -gmm.aic(X)
        elif select_c == 'bic':
            metric = -gmm.bic(X)
    elif select_c == 'valid':
        metric = current_results_dict['best_val_score']
    return metric

def check_convergence(metric, best_metric, n_components, patience, patience_counter):
    if metric > best_metric:
        best_metric = metric
        print(n_components, best_metric)
    else:
        patience_counter += 1
        # might need to add a message here...
        if patience_counter >= patience:
            converged = True
        else:
            converged = False
    return converged, best_metric, patience_counter

def find_best_parameters(n_components, fixed_components, patience, results_dict):
    best_components = n_components
    if not fixed_components:
        best_components -= patience
    # first we need to retrieve the best results
    print(f'Convergence reached at {best_components} components') 
    loss_curves = results_dict[best_components]['all_loss_curves']
    best_seed = results_dict[best_components]['best_seed']
    best_fold_in_init = results_dict[best_components]['best_fold_in_init']            
    all_ws = results_dict[best_components]['all_ws']
    all_ms = results_dict[best_components]['all_ms']
    all_ps = results_dict[best_components]['all_ps']
    w_init = all_ws[best_seed, best_fold_in_init]
    m_init = all_ms[best_seed, best_fold_in_init]
    p_init = all_ps[best_seed, best_fold_in_init] 
    return best_components, best_seed, w_init, m_init, p_init, loss_curves


def calculate_MI(gmm, MI_method='MC', MC_samples=1e5, tol_int=1.49e-8, limit=np.inf):
    if MI_method == 'MC':
        MI = gmm.estimate_MI_MC(MC_samples=MC_samples)
    elif MI_method == 'quad':
        MI = gmm.estimate_MI_quad(tol_int=tol_int, limit=limit)
    return MI


def perform_bootstrap(X, n_bootstrap, n_components, 
                      reg_covar, tol, max_iter, random_state,
                      w_init, m_init, p_init, MC_samples, MI_method='MC'):
    """
    TODO
    """
    MI_estimates = np.zeros(n_bootstrap)
    # bootstrap available samples
    for i in range(n_bootstrap):
        # we use i to change the seed so that the results will be fully reproducible
        rng = np.random.default_rng(i)
        X_bs = rng.choice(X, X.shape[0])
        gmm = single_fit(X_bs, n_components=n_components, reg_covar=reg_covar, 
                    tol=tol, max_iter=max_iter, 
                    random_state=random_state, w_init=w_init, 
                    m_init=m_init, p_init=p_init)
        current_MI_estimate = calculate_MI(gmm, MI_method=MI_method, MC_samples=MC_samples)
        MI_estimates[i] = current_MI_estimate
    MI_mean = np.mean(MI_estimates)
    MI_std = np.sqrt(np.var(MI_estimates, ddof=1))
    return MI_mean, MI_std

              
    
def GMM_MI(X, n_folds=3, n_inits=5, init_type='random_sklearn', reg_covar=1e-15, 
           tol=1e-6, max_iter=10000, max_components=100, select_c='valid', 
           patience=1, bootstrap=True, n_bootstrap=100, fixed_components=False, 
           fixed_components_number=1, MI_method='MC', MC_samples=1e5): 
    """
    Calculate mutual information (MI) distributio on 2D data, using Gaussian mixture models (GMMs).
    The first part performs density estimation of the data using GMMs and k-fold cross-validation.
    The second part uses the fitted model to calculate MI, using either Monte Carlo or quadrature methods.
    The MI uncertainty is calculated through bootstrap.

    Parameters
    ----------
    X : array-like of shape (n_samples, 2)
        Samples from the joint distribution of the two variables whose MI is calculated.
    n_folds : int, default=3
    
    n_inits : int, default=5
    
    init_type='random_sklearn', reg_covar=1e-15, 
           tol=1e-6, max_iter=10000, max_components=100, select_c='valid', 
           patience=1, bootstrap=True, n_bootstrap=100, fixed_components=False, 
           fixed_components_number=1, MI_method='MC', MC_samples=1e5
        
    Returns
    ----------
    MI_mean : float
        Mean of the MI distribution.
    MI_std : float
        Standard deviation of the MI distribution.
    loss_curves : list of lists
        Loss curves of the models trained during cross-validation; only used for debugging.
    """
    converged = False
    best_metric = -np.inf
    patience_counter = 0
    results_dict = {}
    
    assert select_c == 'valid' or select_c == 'aic' or select_c == 'bic', f"select_c must be either 'valid', 'aic' or 'bic, found {select_c}"
    
    for n_components in range(1, max_components+1):
        if fixed_components:
            converged = True
            if n_components < fixed_components_number:
                continue
                
        current_results_dict = cross_validation(X, n_components=n_components, n_folds=n_folds, max_iter=max_iter,
                                       init_type=init_type, n_inits=n_inits, tol=tol, reg_covar=reg_covar)
        results_dict[n_components] = current_results_dict
        
        if not converged:
            metric = select_best_metric(X, current_results_dict=current_results_dict, select_c=select_c,
                                        random_state=current_seed, n_components=n_components, 
                                        init_type=init_type, tol=tol, max_iter=max_iter)
            # check if convergence has been reached based on metric, and save current results
            converged, best_metric, patience_counter = check_convergence(metric=metric, best_metric=best_metric, 
                                                                         n_components=n_components, 
                                                                         patience=patience,
                                                                         patience_counter=patience_counter)
               
        if converged:
            # if we found the optimal number of GMM components, 
            # then we should stop and calculate MI with the previous parameters
            best_components, best_seed, w_init, m_init, p_init, loss_curves = find_best_parameters(n_components=n_components,    
                                                                                                   fixed_components=fixed_components,
                                                                                                   patience=patience, 
                                                                                                   results_dict=results_dict)
            
            if bootstrap:
                MI_mean, MI_std = perform_bootstrap(X, n_bootstrap=n_bootstrap, n_components=best_components, 
                                                    reg_covar=reg_covar, tol=tol, max_iter=max_iter, 
                                                    random_state=best_seed, MC_samples=MC_samples,
                                                    w_init=w_init, m_init=m_init, p_init=p_init)
            else:
                # only now while debugging, we perform a single fit on the entire dataset
                gmm = single_fit(X, n_components=best_components, reg_covar=reg_covar, 
                            tol=tol, max_iter=max_iter, random_state=best_seed, w_init=w_init, 
                            m_init=m_init, p_init=p_init)
                MI_mean = calculate_MI(gmm, MI_method=MI_method, MC_samples=MC_samples)
                MI_std = None
            break
            
    return MI_mean, MI_std, loss_curves