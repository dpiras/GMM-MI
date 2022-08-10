import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from scipy import linalg
import scipy.integrate as integrate
from sklearn.mixture import GaussianMixture as GMM
from sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob, _compute_precision_cholesky, _estimate_gaussian_covariances_full
from sklearn.utils import check_random_state
from sklearn.exceptions import ConvergenceWarning
from gmm_mi.initializations import initialize_parameters
import warnings


class GMM(GMM):
    """
    Custom Gaussian mixture model (GMM) class built on the sklearn GaussianMixture class. 
    Allows to work with a GMM with fixed parameters, without fitting them. 
    It also allows to estimate MI with a certain number of MC samples, 
    or with quadrature method (when in 2D). 
    The different initialization types are dealt with separately, so init_params is not used explicitly.
    We recommend calculating the initial parameters with the provided utilities, and then providing them.
    Most methods and attributes are the same as GaussianMixture; only differences are highlighted here.
    
    Parameters
    ----------
    weights_init : array-like of shape (n_components, ), default=np.array([1.])
        The user-provided initial weights.
        If it is None, a single weight is initialized as 1.
    means_init : array-like of shape (n_components, n_features), default=np.array([0.])
        The user-provided initial means,
        If it is None, a single mean is initialized at 0.
    precisions_init : array-like, default=np.array([1.]).reshape(-1, 1, 1)
        The user-provided initial precisions (inverse of the covariance
        matrices).
        If it is None, a single precision is initialized as 1.
    covariances_init : array-like, default=None
        The user-provided initial covariances (inverse of the precision
        matrices).
        If it is None, covariances are initialized as inverse of initial precision matrices.
    
    Attributes
    ----------
    weights_ : array-like of shape (n_components,)
        The weights of each mixture components. If not fit, set to the initial values.
    means_ : array-like of shape (n_components, n_features)
        The mean of each mixture component. If not fit, set to the initial values.
    covariances_ : array-like
        The covariance of each mixture component. If not fit, set to the initial values.
        The shape depends on `covariance_type`::
            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'
    precisions_ : array-like
        The precision matrices for each component in the mixture. If not fit, set to the initial values.
        A precision matrix is the inverse of a covariance matrix. A covariance matrix is
        symmetric positive definite so the mixture of Gaussian can be
        equivalently parameterized by the precision matrices. Storing the
        precision matrices instead of the covariance matrices makes it more
        efficient to compute the log-likelihood of new samples at test time.
        The shape depends on `covariance_type`::
            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'
    precisions_cholesky_ : array-like
        The Cholesky decomposition of the precision matrices of each mixture component. 
    train_loss : list
        Contains the log-likelihood on training data as a function of iteration.
        Only created and filled if a validation set is provided when fitting the model.
    val_loss : list
        Contains the log-likelihood on validation data as a function of iteration.
        Only created and filled if a validation set is provided when fitting the model.
    """
    def __init__(self,
                 n_components=1,
                 covariance_type="full",
                 tol=1e-3,
                 reg_covar=1e-6,
                 max_iter=100,
                 random_state=None,
                 warm_start=False,
                 verbose=0,
                 verbose_interval=10,
                 weights_init=np.array([1.]),
                 means_init=np.array([0.]),
                 precisions_init=np.array([1.]).reshape(-1, 1, 1),
                 covariances_init=None
                 ):
        super(GMM, self).__init__(n_components=n_components,
                 covariance_type=covariance_type,
                 tol=tol,
                 reg_covar=reg_covar,
                 max_iter=max_iter,
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
        """Compute the log-likelihood of each sample. Removed check_is_fitted function.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row corresponds to a single data point.
        
        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log-likelihood of each sample in `X` under the current model.
        """
        return logsumexp(self._estimate_weighted_log_prob(X), axis=1)

    def predict(self, X):
        """Predict the labels for the data samples in X using trained model.
        Removed check_is_fitted function.
        
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
        Removed check_is_fitted function.
        
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
        """Generate random samples from the Gaussian mixture model.
        Removed check_is_fitted function.

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
        # how many samples for each component?
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
        # this simply expands the information of how many samples per components
        y = np.concatenate(
            [np.full(sample, j, dtype=int) for j, sample in enumerate(n_samples_comp)]
        )
        return (X, y)

    def fit(self, X, y=None, val_set=None):
        """Estimate model parameters with the EM algorithm.
        Unlike the sklearn class, we fit only once and deal with the multiple initialisations elsewhere.
        We add the possibility of recording the log-likelihood on the validation set, if provided.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point. Training set.
        y : Ignored
            Not used, present for API consistency by convention.
        val_set : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point. Validation set.
            
        Returns
        -------
        self : object
            The fitted mixture.
        """
        self.fit_predict(X, y, val_set)
        return self
    
    def fit_predict(self, X, y=None, val_set=None):
        """Estimate model parameters using X and predict the labels for X.
        The method fits the model once times and sets the parameters with
        which the model has the largest likelihood. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
        raised.
        Unlike the sklearn class, we fit only once and deal with the multiple initialisations elsewhere.
        We add the possibility of recording the log-likelihood on the validation set, if provided.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point. Training set.
        y : Ignored
            Not used, present for API consistency by convention.
        val_set : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point. Validation set.
            
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

        do_init = True
        n_init = 1
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

    def score_samples_marginal(self, X, index):
        """Compute the log-likelihood of each sample for the marginal model, 
        indexed by either 0 (x) or 1 (y).
        
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
            Log-likelihood of each sample in X under the marginal model.
        """
        # in 1-D the Cholesky decomposition is simply the inverse sqrt of the variance
        oned_cholesky = np.sqrt(1/self.covariances_[:, index, index]).reshape(-1, 1, 1)
        marginal_logprob = _estimate_log_gaussian_prob(
            X, self.means_[:, index].reshape(-1, 1), oned_cholesky, self.covariance_type
        )
        return logsumexp(np.log(self.weights_) + marginal_logprob, axis=1)

    def estimate_MI_MC(self, MC_samples=1e5):
        """Compute the mutual information (MI) associated with a particular GMM model, 
        using MC integration.
        
        Parameters
        ----------
        MC_samples : integer, default=1e5
            Number of Monte Carlo (MC) samples to perform numerical integration of the MI integral.
        
        Returns
        -------
        MI : float
            The value of mutual information.
        """
        points, _ = self.sample(MC_samples)       
        # evaluate the log-likelihood for the joint probability
        joint = self.score_samples(points)
        # and the marginals; index=0 corresponds to x, index=y corresponds to y
        marginal_x = self.score_samples_marginal(points[:, :1], index=0)
        marginal_y = self.score_samples_marginal(points[:, 1:], index=1)
        MI = np.mean(joint - marginal_x - marginal_y)
        self.MI = MI
        return MI

    def estimate_MI_quad(self, tol_int=1.49e-8, limit=np.inf):
        """Compute the mutual information (MI) associated with a particular GMM model, 
        using quadrature integration.
        
        Parameters
        ----------
        tol_int : float, default=1.49e-8
            Integral tolerance; the default value is the one form scipy. 
        limit : float, default=np.inf
            The extrema of the integral to calculate. Usually the whole plane, so defaults to inf.
            Integral goes from -limit to +limit.
        
        Returns
        -------
        MI : float
            The value of mutual information.
        """
        # we create a GMM object to pass to the integral functions
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
        self.MI = MI
        return MI
    
    
def log_pdf(y, x, model):
    """Log-likelihood in 2D for a given model
    (typically a Gaussian mixture model, GMM).
        
    Parameters
    ----------
    y : float
        The y variable.
    x : float, 
        The x variable.
    model : instance of class with score_samples method
        The model whose log-likelihood we calculate; typically a GMM.
    
    Returns
    -------
    integrand : float
        The value of the integrand.
    """
    y = np.array(y)
    x = np.array(x)
    X = np.concatenate((y.reshape(1, 1), x.reshape(1, 1))).T
    return model.score_samples(X)


def entropy_2d_integrand(y, x, model):
    """Integrand function of 2D entropy for a given model
    (typically a Gaussian mixture model, GMM).
        
    Parameters
    ----------
    y : float
        The y variable.
    x : float, 
        The x variable.
    model : instance of class with score_samples method
        The model whose entropy we calculate; typically a GMM.
    
    Returns
    -------
    integrand : float
        The value of the integrand.
    """
    logp = log_pdf(y, x, model)
    p = np.exp(logp)
    integrand = p*logp
    return integrand


def entropy_1d_integrand(x, model, index):
    """Integrand function of 1D entropy for a given model
    (typically a Gaussian mixture model, GMM).
        
    Parameters
    ----------
    x : float, 
        The x variable.
    model : instance of class with score_samples method
        The model whose entropy we calculate; typically a GMM.
        Must have a score_samples_marginal method.
    index : int
        The index used to calculate entropy, either 0 (x) or 1 (y).
    
    Returns
    -------
    integrand : float
        The value of the integrand.
    """
    assert index == 0 or index == 1, f"Index must be either 0 (x) or 1 (y); found '{index}'"
    x = np.array(x).reshape(1, 1)
    logp = model.score_samples_marginal(x, index)
    p = np.exp(logp)
    integrand = p*logp
    return integrand
    