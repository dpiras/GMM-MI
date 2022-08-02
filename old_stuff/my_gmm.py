import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from scipy import linalg
from sklearn.mixture import GaussianMixture as GMM
from sklearn.mixture import BayesianGaussianMixture as BGMM
import numbers
import gc

def _estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type):
        """Estimate the log Gaussian probability.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        means : array-like of shape (n_components, n_features)
        precisions_chol : array-like
            Cholesky decompositions of the precision matrices.
            'full' : shape of (n_components, n_features, n_features)
            'tied' : shape of (n_features, n_features)
            'diag' : shape of (n_components, n_features)
            'spherical' : shape of (n_components,)
        covariance_type : {'full', 'tied', 'diag', 'spherical'}
        Returns
        -------
        log_prob : array, shape (n_samples, n_components)
        """
        n_samples, n_features = X.shape
        n_components, _ = means.shape
        # det(precision_chol) is half of det(precision)
        log_det = _compute_log_det_cholesky(precisions_chol, covariance_type, n_features)

        if covariance_type == "full":
            log_prob = np.empty((n_samples, n_components))
            for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
                y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
                log_prob[:, k] = np.sum(np.square(y), axis=1)

        elif covariance_type == "tied":
            log_prob = np.empty((n_samples, n_components))
            for k, mu in enumerate(means):
                y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
                log_prob[:, k] = np.sum(np.square(y), axis=1)

        elif covariance_type == "diag":
            precisions = precisions_chol ** 2
            log_prob = (
                np.sum((means ** 2 * precisions), 1)
                - 2.0 * np.dot(X, (means * precisions).T)
                + np.dot(X ** 2, precisions.T)
            )

        elif covariance_type == "spherical":
            precisions = precisions_chol ** 2
            log_prob = (
                np.sum(means ** 2, 1) * precisions
                - 2 * np.dot(X, means.T * precisions)
                + np.outer(row_norms(X, squared=True), precisions)
            )
        return -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det
    
def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )

def _compute_precision_cholesky(covariances, covariance_type):
    """Compute the Cholesky decomposition of the precisions.
    Parameters
    ----------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.
    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
    """
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar."
    )

    if covariance_type == "full":
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = linalg.cholesky(covariance, lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = linalg.solve_triangular(
                cov_chol, np.eye(n_features), lower=True
            ).T
    elif covariance_type == "tied":
        _, n_features = covariances.shape
        try:
            cov_chol = linalg.cholesky(covariances, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol = linalg.solve_triangular(
            cov_chol, np.eye(n_features), lower=True
        ).T
    else:
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1.0 / np.sqrt(covariances)
    return precisions_chol


def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    """Compute the log-det of the cholesky decomposition of matrices.
    Parameters
    ----------
    matrix_chol : array-like
        Cholesky decompositions of the matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
    n_features : int
        Number of features.
    Returns
    -------
    log_det_precision_chol : array-like of shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    if covariance_type == "full":
        n_components, _, _ = matrix_chol.shape
        log_det_chol = np.sum(
            np.log(matrix_chol.reshape(n_components, -1)[:, :: n_features + 1]), 1
        )

    elif covariance_type == "tied":
        log_det_chol = np.sum(np.log(np.diag(matrix_chol)))

    elif covariance_type == "diag":
        log_det_chol = np.sum(np.log(matrix_chol), axis=1)

    else:
        log_det_chol = n_features * (np.log(matrix_chol))

    return log_det_chol



class my_GMM():
    def __init__(
        self,
        means,
        covariances, 
        weights,
        n_components=1,
        random_state=None,
    ):
        self.means_ = means
        self.covariances_ = covariances
        self.weights_ = weights
        self.n_components = n_components
        self.random_state = random_state
        self.covariance_type = 'full' # hardcoded for now
        self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type
            )
        
        
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

        X = np.vstack(
            [
                rng.multivariate_normal(mean, covariance, int(sample))
                for (mean, covariance, sample) in zip(
                    self.means_, self.covariances_, n_samples_comp
                )
            ]
        )
        
        y = np.concatenate(
            [np.full(sample, j, dtype=int) for j, sample in enumerate(n_samples_comp)]
        )

        return (X, y)
    



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
        #X = self._validate_data(X, reset=False)

        return logsumexp(self._estimate_weighted_log_prob(X), axis=1)
    
    def _estimate_weighted_log_prob(self, X):
        """Estimate the weighted log-probabilities, log P(X | Z) + log weights.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        weighted_log_prob : array, shape (n_samples, n_component)
        """
        return self._estimate_log_prob(X) + self._estimate_log_weights()
    
    def _estimate_log_prob(self, X):
        return _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type
        )
    
    def _estimate_log_weights(self):
        return np.log(self.weights_)
    
    
    
    
    
# here we define a custom model for a fixed gmm with known parameters
class FixedParMixture:
    """ A model to estimate gaussian mixture with fixed parameters matrix. 
        This is only needed to estimate its log-likelihood, not to sample from it
        Note this is only for a 1D GMM, as in n_features=1! 
    """
    def __init__(self, n_components, mean, cov, weight):
        self.n_components = n_components
        self.mean = mean
        self.cov = cov
        self.w = weight

    
    def estimate_prob(self, X):
        marginal = 0
        for i in range(self.n_components):
            mu = self.mean[i]
            sigma = self.cov[i]
            w = self.w[i]
            marginal += w*multivariate_normal.pdf(x=np.array([X]).T, mean=np.array([mu]), cov=np.array([sigma]))         
        return marginal
        
    
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
            
        This is taken from the sklearn source code
        """

        return np.log(self.estimate_prob(X))

def marginal_loglike(gmm, points, n_components, i):
    ll = FixedParMixture(n_components, gmm.means_[:, i], gmm.covariances_[:, i, i], gmm.weights_).score_samples(points[:, i])
    return ll

def estimate_MI_single_posterior_sample(gmm, MC_samples=100):
    # we need to sample from the joint distribution first
    n_components = gmm.n_components
    points, clusters = gmm.sample(MC_samples)
    # we can already evaluate the log-likelihood for the joint probability
    joint = gmm.score_samples(points)
    # we now need to define the marginals
    marginal_x = marginal_loglike(gmm, points, n_components, 0)
    marginal_y = marginal_loglike(gmm, points, n_components, 1)
    return np.mean(joint - marginal_x - marginal_y)

class FixedParMixture_pygmmis:
    """ A model to estimate gaussian mixture with fixed parameters matrix. 
        This is only needed to estimate its log-likelihood, not to sample from it
        Note this is only for a 1D GMM, as in n_features=1! 
    """
    def __init__(self, n_components, mean, cov, weight):
        self.n_components = n_components
        self.mean = mean
        self.cov = cov
        self.w = weight

    
    def estimate_prob(self, X):
        marginal = 0
        for i in range(self.n_components):
            mu = self.mean[i]
            sigma = self.cov[i]
            w = self.w[i]
            marginal += w*multivariate_normal.pdf(x=np.array([X]).T, mean=np.array([mu]), cov=np.array([sigma]))         
        return marginal
        
    
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
            
        This is taken from the sklearn source code
        """

        return np.log(self.estimate_prob(X))

def marginal_loglike_pygmmis(gmm, points, n_components, i):
    ll = FixedParMixture_pygmmis(n_components, gmm.mean[:, i], gmm.covar[:, i, i], gmm.amp).score_samples(points[:, i])
    return ll

def estimate_MI_single_posterior_sample_pygmmis(gmm, MC_samples=100):
    # we need to sample from the joint distribution first
    n_components = gmm.K#gmm.n_components
    points = gmm.draw(int(MC_samples))#gmm.sample(MC_samples)
    # we can already evaluate the log-likelihood for the joint probability
    joint = gmm.logL(points)
    # we now need to define the marginals
    marginal_x = marginal_loglike_pygmmis(gmm, points, n_components, 0)
    marginal_y = marginal_loglike_pygmmis(gmm, points, n_components, 1)
    return np.mean(joint - marginal_x - marginal_y)

def consistent(a, b, sa, sb, C=1):
    return np.abs(a-b) / np.sqrt(sa**2+sb**2) <= C

def bootstrap_multiprocessing(i, k, X, X_valid, reg_covar, MC_samples, BGMM_flag=False):
        # draw bootstrap data
        rng = np.random.default_rng(seed=i)
        X_bs = rng.choice(X, X.shape[0])
        # fit GMM to current data
        if BGMM_flag:
            gmm = BGMM(n_components=k, reg_covar=reg_covar, weight_concentration_prior=20, random_state=i, max_iter=200).fit(X_bs)
        else:
            gmm = GMM(n_components=k, reg_covar=reg_covar, random_state=i).fit(X_bs)
        # evaluate ALL on both current bs of training data and fixed validation data
        # estimate MI for current set of parameters
        MI_MC_estimate = estimate_MI_single_posterior_sample(gmm, MC_samples=MC_samples)
        return MI_MC_estimate, gmm.score(X), gmm.score(X_valid)