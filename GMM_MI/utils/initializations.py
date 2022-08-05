import numpy as np
from sklearn.utils import check_random_state
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_covariances_full
from scipy.special import gamma
from sklearn import cluster
from sklearn.cluster import kmeans_plusplus

def calculate_scale(X, n_components, n_dim):
    min_pos = X.min(axis=0)
    max_pos = X.max(axis=0)
    vol_data = np.prod(max_pos-min_pos)
    scale = (vol_data / n_components * gamma(n_dim*0.5 + 1))**(1/n_dim) / np.sqrt(np.pi)
    return scale
    
    
def random_initialization(X, n_components, n_dim, n_samples, scale, random_state):
    weights = np.repeat(1/n_components, n_components)
    # initialize components around data points with uncertainty s
    refs = random_state.randint(0, n_samples, size=n_components)
    means = X[refs] + random_state.multivariate_normal(np.zeros(n_dim), scale**2 * np.eye(n_dim), size=n_components)
    covariances = np.repeat(scale**2 * np.eye(n_dim)[np.newaxis, :, :], n_components, axis=0)
    return weights, means, covariances
 
    
def minmax_initialization(X, n_components, n_dim, scale, random_state):
    weights = np.repeat(1/n_components, n_components)
    min_pos = X.min(axis=0)
    max_pos = X.max(axis=0)
    means = min_pos + (max_pos-min_pos)*random_state.rand(n_components, n_dim)
    covariances = np.repeat(scale**2 * np.eye(n_dim)[np.newaxis, :, :], n_components, axis=0)
    return weights, means, covariances


def kmeans_initialization(X, n_components, n_dim, random_state):
    from scipy.cluster.vq import kmeans2
    center, label = kmeans2(X, n_components, seed=random_state)
    weights = np.zeros(n_components)
    means = np.zeros((n_components, n_dim))
    covariances = np.zeros((n_components, n_dim, n_dim))

    for k in range(n_components):
        mask = (label == k)
        weights[k] = mask.sum() / len(X)
        means[k,:] = X[mask].mean(axis=0)
        d_m = X[mask] - means[k,:] 
        # funny way of saying: for each point i, do the outer product
        # of d_m with its transpose and sum over i
        covariances[k,:,:] = (d_m[:, :, None] * d_m[:, None, :]).sum(axis=0) / len(X)
    return weights, means, covariances
    
def calculate_parameters_from_resp(resp, X, n_samples, reg_covar=0):
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    weights = nk/n_samples
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    # the regularization added to the diagonal of the covariance matrices (to prevent singular covariance matrices).
    # set to 0 as we are only initializing here
    covariances = _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar=reg_covar)        
    return weights, means, covariances


def random_sklearn_initialization(X, n_components, n_samples, random_state, reg_covar=0):
    resp = random_state.rand(n_samples, n_components)
    resp /= resp.sum(axis=1)[:, np.newaxis]
    weights, means, covariances = calculate_parameters_from_resp(resp, X, n_samples, reg_covar=0)    
    return weights, means, covariances


def kmeans_sklearn_initialization(X, n_components, n_samples, random_state, reg_covar=0):
    resp = np.zeros((n_samples, n_components))
    label = (
        cluster.KMeans(
            n_clusters=n_components, n_init=1, random_state=random_state
        )
        .fit(X)
        .labels_
    )
    resp[np.arange(n_samples), label] = 1
    weights, means, covariances = calculate_parameters_from_resp(resp, X, n_samples, reg_covar=0)    
    return weights, means, covariances


def initialize_parameters(X, random_state=None, n_components=1, init_type='random_sklearn', scale=None):
    """Initialize the GMM model parameters (weights, means and covariances).
    
    Parameters
    ----------
    X : array-like of shape  (n_samples, n_features)
        The data based on which the GMM initialization is calculated.    
    random_state : int, default=None
        A random seed used for the method chosen to initialize the parameters.
    n_components: int, default=1
        Number of components of the GMM to initialize.
    init_type : {'random', 'minmax', 'kmeans', 'random_sklearn', 'kmeans_sklearn'}, default='random_sklearn'
        The method used to initialize the weights, the means and the precisions.
        Must be one of:
            'random': weights are set uniformly, covariances are proportional to identity (with prefactor s^2). 
             For each mean, a data sample is selected at random, and a multivariate Gaussian with variance s^2 offset is added.
            'minmax': same as above, but means are distributed randomly over the range that is covered by data.
            'kmeans': k-means clustering run as in Algorithm 1 from Bloemer & Bujna (arXiv:1312.5946), as implemented by Melchior & Goulding (arXiv:1611.05806)
             WARNING: The result of this call are not deterministic even if rng is set because scipy.cluster.vq.kmeans2 uses its own initialization. 
             TO DO: require scipy > 1.7, and include "seed=random_state" in the kmeans call
            'kmeans_sklearn' : responsibilities are initialized using kmeans.
            'random_sklearn' : responsibilities are initialized randomly.
            Note we are not including the most recent sklearn initializations.
    scale : float, default=None
        If set, sets component variances in the 'random' and 'minmax' cases. 
        If s is not given, it will be set such that the volume of all components completely fills the space covered by data.
        
    Returns
    ----------
    weights : array, shape (n_components, 1)
        The initial weights of the GMM model.
    means : array, shape (n_components, n_features)
        The initial means of the GMM model.        
    covariances : array, shape (n_components, n_features, n_features)
        The initial covariance matrices of the GMM model.        
    """
    n_samples, n_dim = X.shape
    random_state = check_random_state(random_state)
    
    if scale is None and (init_type=='random' or init_type=='minmax'):
        scale = calculate_scale(X, n_components, n_dim) 
        
    if init_type == "random":
        weights, means, covariances = random_initialization(X, n_components, n_dim, n_samples, scale, random_state)
    elif init_type == "minmax":
        weights, means, covariances = minmax_initialization(X, n_components, n_dim, scale, random_state)
    elif init_type == 'kmeans':
        weights, means, covariances = kmeans_initialization(X, n_components, n_dim, random_state)
    elif init_type == "random_sklearn":
        weights, means, covariances = random_sklearn_initialization(X, n_components, n_samples, random_state)
    elif init_type == "kmeans_sklearn":
         weights, means, covariances = kmeans_sklearn_initialization(X, n_components, n_samples, random_state)
    else:
        raise ValueError(f"Initialisation type not known. It should be one of "
                         f"'random', 'minmax', 'kmeans', 'random_sklearn', 'kmeans_sklearn'; found '{init_type}'") 

    # all matrices can be inverted at once
    precisions = np.linalg.inv(covariances)
    return weights, means, covariances, precisions
