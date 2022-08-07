import numpy as np
from sklearn.utils import check_random_state
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_covariances_full
from sklearn import cluster
from scipy.special import gamma


def calculate_scale(X, n_components):
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
    _, n_features = X.shape
    min_pos = X.min(axis=0)
    max_pos = X.max(axis=0)
    vol_data = np.prod(max_pos-min_pos)
    # radius of n-ball given its volume; see https://en.wikipedia.org/wiki/Volume_of_an_n-ball
    scale = (vol_data / n_components * gamma(n_features*0.5 + 1))**(1/n_features) / np.sqrt(np.pi)
    return scale
    
    
def random_initialization(X, n_components, scale, random_state):
    """Calculate GMM parameters randomly. See initialize_parameters for more details.
    
    Parameters
    ----------
    X : array-like of shape  (n_samples, n_features)
        The data based on which the GMM initialization is calculated.    
    n_components : int
        Number of GMM components.
    scale : float
        Scale prefactor, to be used to initialize the GMM.
    random_state : RandomState instance
        Used to sample the means.
        
    Returns
    ----------
    weights : array, shape (n_components, 1)
        The initial weights of the GMM model.
    means : array, shape (n_components, n_features)
        The initial means of the GMM model.        
    covariances : array, shape (n_components, n_features, n_features)
        The initial covariance matrices of the GMM model.        
    """
    n_samples, n_features = X.shape
    weights = np.repeat(1/n_components, n_components)
    # initialize components around data points with uncertainty s
    refs = random_state.randint(0, n_samples, size=n_components)
    means = X[refs] + random_state.multivariate_normal(np.zeros(n_features), scale**2 * np.eye(n_features), size=n_components)
    covariances = np.repeat(scale**2 * np.eye(n_features)[np.newaxis, :, :], n_components, axis=0)
    return weights, means, covariances
 
    
def minmax_initialization(X, n_components, scale, random_state):
    """Calculate GMM parameters randomly, but with the means sampeld between the minimum and maximum value of the data. 
    See initialize_parameters for more details.
    
    Parameters
    ----------
    X : array-like of shape  (n_samples, n_features)
        The data based on which the GMM initialization is calculated.    
    n_components : int
        Number of GMM components.
    scale : float
        Scale prefactor, to be used to initialize the GMM.
    random_state : RandomState instance
        Used to sample the means.
        
    Returns
    ----------
    weights : array, shape (n_components, 1)
        The initial weights of the GMM model.
    means : array, shape (n_components, n_features)
        The initial means of the GMM model.        
    covariances : array, shape (n_components, n_features, n_features)
        The initial covariance matrices of the GMM model.        
    """
    _, n_features = X.shape
    weights = np.repeat(1/n_components, n_components)
    min_pos = X.min(axis=0)
    max_pos = X.max(axis=0)
    means = min_pos + (max_pos-min_pos)*random_state.rand(n_components, n_features)
    covariances = np.repeat(scale**2 * np.eye(n_features)[np.newaxis, :, :], n_components, axis=0)
    return weights, means, covariances


def kmeans_initialization(X, n_components, random_state):
    """Calculate GMM parameters based on Bloemer & Bujna (arXiv:1312.5946). 
    See initialize_parameters for more details.
    
    Parameters
    ----------
    X : array-like of shape  (n_samples, n_features)
        The data based on which the GMM initialization is calculated.    
    n_components : int
        Number of GMM components.
    random_state : RandomState instance
        Used to initialize k-means.
        
    Returns
    ----------
    weights : array, shape (n_components, 1)
        The initial weights of the GMM model.
    means : array, shape (n_components, n_features)
        The initial means of the GMM model.        
    covariances : array, shape (n_components, n_features, n_features)
        The initial covariance matrices of the GMM model.        
    """
    _, n_features = X.shape
    from scipy.cluster.vq import kmeans2
    _, label = kmeans2(X, n_components, seed=random_state)
    weights = np.zeros(n_components)
    means = np.zeros((n_components, n_features))
    covariances = np.zeros((n_components, n_features, n_features))

    for k in range(n_components):
        mask = (label == k)
        weights[k] = mask.sum() / len(X)
        means[k,:] = X[mask].mean(axis=0)
        d_m = X[mask] - means[k,:] 
        # funny way of saying: for each point i, do the outer product of d_m with its transpose and sum over i
        # in other words, simply the definition of sample covariance matrix
        covariances[k,:,:] = (d_m[:, :, None] * d_m[:, None, :]).sum(axis=0) / len(X)
    return weights, means, covariances
    
    
def calculate_parameters_from_responsibilities(responsibilities, X, reg_covar=0):
    """Calculate GMM parameters given the responsibilities of each sample. 
    The responsibilities are the probabilities that each sample belongs to each component.
    Weights, means and covariances are calculated as in an M-step.
    See initialize_parameters for more details.
    
    Parameters
    ----------
    responsibilities : array-like of shape  (n_samples, n_components)
        The responsibilities.   
    X : array-like of shape  (n_samples, n_features)
        The data based on which the GMM initialization is calculated.  
    reg_covar : float, default=0
        Constant regularisation term added to the diagonal of each covariance matrix, to avoid singular matrices.
        
    Returns
    ----------
    weights : array, shape (n_components, 1)
        The initial weights of the GMM model.
    means : array, shape (n_components, n_features)
        The initial means of the GMM model.        
    covariances : array, shape (n_components, n_features, n_features)
        The initial covariance matrices of the GMM model.        
    """
    n_samples, _ = X.shape
    # we add a small epsilon to avoid underflows
    nk = responsibilities.sum(axis=0) + 10 * np.finfo(responsibilities.dtype).eps
    weights = nk/n_samples
    means = np.dot(responsibilities.T, X) / nk[:, np.newaxis]
    # the regularization added to the diagonal of the covariance matrices (to prevent singular covariance matrices);
    covariances = _estimate_gaussian_covariances_full(responsibilities, X, nk, means, reg_covar=reg_covar)        
    return weights, means, covariances


def random_sklearn_initialization(X, n_components, random_state, reg_covar=0):
    """Calculate GMM parameters by assigning the responsibilities randomly.
    This is the default initialization, and is taken from sklearn.
    See initialize_parameters for more details.
    
    Parameters
    ---------- 
    X : array-like of shape  (n_samples, n_features)
        The data based on which the GMM initialization is calculated.    
    n_components : int
        Number of GMM components.
    random_state : RandomState instance
        Used to sample the means. 
    reg_covar : float, default=0
        Constant regularisation term added to the diagonal of each covariance matrix, to avoid singular matrices.
        
    Returns
    ----------
    weights : array, shape (n_components, 1)
        The initial weights of the GMM model.
    means : array, shape (n_components, n_features)
        The initial means of the GMM model.        
    covariances : array, shape (n_components, n_features, n_features)
        The initial covariance matrices of the GMM model.        
    """
    n_samples, _ = X.shape
    responsibilities = random_state.rand(n_samples, n_components)
    responsibilities /= responsibilities.sum(axis=1)[:, np.newaxis]
    weights, means, covariances = calculate_parameters_from_responsibilities(responsibilities, X, reg_covar=reg_covar)    
    return weights, means, covariances


def kmeans_sklearn_initialization(X, n_components, random_state, reg_covar=0):
    """Calculate GMM parameters by assigning the responsibilities via k-means.
    Note that in this case each point is assigned with probability 1 to a single cluster.
    Taken from sklearn. See initialize_parameters for more details.
    
    Parameters
    ---------- 
    X : array-like of shape  (n_samples, n_features)
        The data based on which the GMM initialization is calculated.    
    n_components : int
        Number of GMM components.
    random_state : RandomState instance
        Used to sample the means. 
    reg_covar : float, default=0
        Constant regularisation term added to the diagonal of each covariance matrix, to avoid singular matrices.
        
    Returns
    ----------
    weights : array, shape (n_components, 1)
        The initial weights of the GMM model.
    means : array, shape (n_components, n_features)
        The initial means of the GMM model.        
    covariances : array, shape (n_components, n_features, n_features)
        The initial covariance matrices of the GMM model.        
    """
    n_samples, _ = X.shape
    responsibilities = np.zeros((n_samples, n_components))
    label = (
        cluster.KMeans(
            n_clusters=n_components, n_init=1, random_state=random_state
        )
        .fit(X)
        .labels_
    )
    responsibilities[np.arange(n_samples), label] = 1
    weights, means, covariances = calculate_parameters_from_responsibilities(responsibilities, X, reg_covar=reg_covar)    
    return weights, means, covariances


def initialize_parameters(X, random_state=None, n_components=1, init_type='random_sklearn', scale=None, reg_covar=0):
    """Initialize the Guassian mixture model (GMM) parameters (weights, means, covariances and precisions).
    
    Parameters
    ----------
    X : array-like of shape  (n_samples, n_features)
        The data based on which the GMM initialization is calculated.    
    random_state : int, default=None
        A random seed used for the method chosen to initialize the parameters.
    n_components : int, default=1
        Number of GMM components to initialize.
    init_type : {'random', 'minmax', 'kmeans', 'random_sklearn', 'kmeans_sklearn'}, default='random_sklearn'
        The method used to initialize the weights, the means, the covariances and the precisions.
        Must be one of:
            'random': weights are set uniformly, covariances are proportional to identity (with prefactor s^2). 
            For each mean, a data sample is selected at random, and a multivariate Gaussian with variance s^2 offset is added.
            'minmax': same as above, but means are distributed randomly over the range that is covered by data.
            'kmeans': k-means clustering run as in Algorithm 1 from Bloemer & Bujna (arXiv:1312.5946), 
            as implemented by Melchior & Goulding (arXiv:1611.05806).
            'random_sklearn': responsibilities are initialized randomly, i.e. every point is randomly assigned to a component.
            Weights, means and covariances are then calculated by performing an M-step on these responsibilities:
            this means that weights are calculated as the average probability that a sample belongs to each components, while
            means and covariance matrices are the weighted mean and covariance of the samples.
            'kmeans_sklearn': responsibilities are initialized using k-means, as implemented by sklearn. 
            The rest is as in 'random_sklearn'.
            Note we are currently not including the most recent sklearn initializations.
    scale : float, default=None
        If set, sets component variances in the 'random' and 'minmax' cases. 
        If scale is not given, it will be set such that the volume of all components completely fills the space covered by data.
    TODO: add possibility to add reg_covar to covariances
    
    Returns
    ----------
    weights : array, shape (n_components, 1)
        The initial weights of the GMM model.
    means : array, shape (n_components, n_features)
        The initial means of the GMM model.        
    covariances : array, shape (n_components, n_features, n_features)
        The initial covariance matrices of the GMM model.  
    precisions : array, shape (n_components, n_features, n_features)
        The initial precision matrices of the GMM model (inverse of covariance).  
    """
    random_state = check_random_state(random_state)
    
    if scale is None and (init_type=='random' or init_type=='minmax'):
        scale = calculate_scale(X, n_components) 
        
    if init_type == "random":
        weights, means, covariances = random_initialization(X, n_components, scale, random_state)
    elif init_type == "minmax":
        weights, means, covariances = minmax_initialization(X, n_components, scale, random_state)
    elif init_type == 'kmeans':
        weights, means, covariances = kmeans_initialization(X, n_components, random_state)
    elif init_type == "random_sklearn":
        weights, means, covariances = random_sklearn_initialization(X, n_components, random_state)
    elif init_type == "kmeans_sklearn":
         weights, means, covariances = kmeans_sklearn_initialization(X, n_components, random_state)
    else:
        raise ValueError(f"Initialisation type not known. It should be one of "
                         f"'random', 'minmax', 'kmeans', 'random_sklearn', 'kmeans_sklearn'; found '{init_type}'") 

    # all matrices can be inverted at once
    precisions = np.linalg.inv(covariances)
    return weights, means, covariances, precisions
