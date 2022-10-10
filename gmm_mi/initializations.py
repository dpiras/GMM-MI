import numpy as np
from sklearn.utils import check_random_state
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_covariances_full
from sklearn import cluster
from scipy.special import gamma


class Inits:
    """Generic class to deal with the different initializations. See initialize_parameters 
    for more details, and for all parameters that need to be passed.
    """
    def __init__(self, X, random_state=None, n_components=1, 
                 init_type='random_sklearn', scale=None, reg_covar=0):
        self.X = X
        self.n_samples, self.n_features = self.X.shape
        self.random_state = check_random_state(random_state)
        self.n_components = n_components
        self.init_type = init_type
        if scale is None and (self.init_type=='random' or self.init_type=='minmax'):
            self.scale = self.calculate_scale() 
        elif scale is None and self.init_type=='randomized_kmeans':
            self.scale = 1.0
        else:
            self.scale = scale
        self.reg_covar = reg_covar             

    def calculate_scale(self):
        """Calculate scale prefactor ('s') for the 'random' and 'minmax' initializations.

        Returns
        ----------
        scale : float
            Scale prefactor, to be used to initialize the GMM.
        """
        min_pos = self.X.min(axis=0)
        max_pos = self.X.max(axis=0)
        vol_data = np.prod(max_pos-min_pos)
        # radius of n-ball given its volume; see https://en.wikipedia.org/wiki/Volume_of_an_n-ball
        scale = (vol_data / self.n_components * gamma(self.n_features*0.5 + 1))**(1/self.n_features) / np.sqrt(np.pi)
        return scale
    
    def calculate_parameters_from_responsibilities(self, responsibilities):
        """Calculate GMM parameters given the responsibilities of each sample. 
        The responsibilities are the probabilities that each sample belongs to each component.
        Weights, means and covariances are calculated as in an M-step.

        Parameters
        ----------
        responsibilities : array-like of shape (n_samples, n_components)
            The responsibilities.     

        Returns
        ----------
        weights : array, shape (n_components, 1)
            The initial weights of the GMM model.
        means : array, shape (n_components, n_features)
            The initial means of the GMM model.        
        covariances : array, shape (n_components, n_features, n_features)
            The initial covariance matrices of the GMM model.        
        """
        # we add a small epsilon to avoid underflows
        nk = responsibilities.sum(axis=0) + 10 * np.finfo(responsibilities.dtype).eps
        weights = nk/self.n_samples
        means = np.dot(responsibilities.T, self.X) / nk[:, np.newaxis]
        # we don't add any regularization  to the diagonal of the covariance matrices here
        covariances = _estimate_gaussian_covariances_full(responsibilities, self.X, nk, means, reg_covar=0)        
        return weights, means, covariances
    
    def add_regularization(self, covariances):
        """Add regularization to the diagonal of the covariance matrices, to avoid singularities.
        
        Parameters
        ----------
        covariances : array, shape (n_components, n_features, n_features)
            The unregularized covariance matrices of the GMM model.  

        Returns
        ---------- 
        covariances : array, shape (n_components, n_features, n_features)
            The regularized covariance matrices of the GMM model.  
        """
        regularization = np.repeat((self.reg_covar*np.eye(self.n_features))[np.newaxis, :, :], self.n_components, axis=0)
        covariances += regularization   
        return covariances   
    
    
class RandomInit(Inits):
    def calculate_parameters(self):
        weights = np.repeat(1/self.n_components, self.n_components)
        # initialize components around data points with uncertainty s
        refs = self.random_state.randint(0, self.n_samples, size=self.n_components)
        means = self.X[refs] + self.random_state.multivariate_normal(np.zeros(self.n_features), 
                                                                     self.scale**2 * np.eye(self.n_features), 
                                                                     size=self.n_components)
        covariances = np.repeat(self.scale**2 * np.eye(self.n_features)[np.newaxis, :, :], self.n_components, axis=0)
        return weights, means, covariances

    
class KmeansInit(Inits):
    def calculate_parameters(self):
        from scipy.cluster.vq import kmeans2
        _, label = kmeans2(self.X, self.n_components, seed=self.random_state)
        weights = np.zeros(self.n_components)
        means = np.zeros((self.n_components, self.n_features))
        covariances = np.zeros((self.n_components, self.n_features, self.n_features))
        for k in range(self.n_components):
            mask = (label == k)
            weights[k] = mask.sum() / len(self.X)
            means[k,:] = self.X[mask].mean(axis=0)
            d_m = self.X[mask] - means[k,:] 
            # funny way of saying: for each point i, do the outer product of d_m with its transpose and sum over i
            # in other words, simply the definition of sample covariance matrix
            covariances[k,:,:] = (d_m[:, :, None] * d_m[:, None, :]).sum(axis=0) / len(self.X)
        return weights, means, covariances
    
    
class RandomizedKmeansInit(Inits):
    def calculate_parameters(self):
        from scipy.cluster.vq import kmeans2
        _, label = kmeans2(self.X, self.n_components, seed=self.random_state)
        weights = np.zeros(self.n_components)
        means = np.zeros((self.n_components, self.n_features))
        covariances = np.zeros((self.n_components, self.n_features, self.n_features))
        for k in range(self.n_components):
            mask = (label == k)
            weights[k] = mask.sum() / len(self.X)
            means[k,:] = self.X[mask].mean(axis=0)
            d_m = self.X[mask] - means[k,:] 
            # funny way of saying: for each point i, do the outer product of d_m with its transpose and sum over i
            # in other words, simply the definition of sample covariance matrix
            covariances[k,:,:] = (d_m[:, :, None] * d_m[:, None, :]).sum(axis=0) / len(self.X)           
        # after running k-means, we push the means according to their covariance;
        # this should reduce the risks of getting stuck in local optima
        for k in range(self.n_components):
            means[k,:] = means[k,:] + self.scale*self.random_state.multivariate_normal(np.zeros(self.n_features), 
                                                                     covariances[k,:,:], size=1)
        return weights, means, covariances    
    

class MinMaxInit(Inits):
    def calculate_parameters(self):
        weights = np.repeat(1/self.n_components, self.n_components)
        min_pos = self.X.min(axis=0)
        max_pos = self.X.max(axis=0)
        means = min_pos + (max_pos-min_pos)*self.random_state.rand(self.n_components, self.n_features)
        covariances = np.repeat(self.scale**2 * np.eye(self.n_features)[np.newaxis, :, :], self.n_components, axis=0)
        return weights, means, covariances

    
class RandomSklearnInit(Inits):
    def calculate_parameters(self):
        responsibilities = self.random_state.rand(self.n_samples, self.n_components)
        responsibilities /= responsibilities.sum(axis=1)[:, np.newaxis]
        weights, means, covariances = self.calculate_parameters_from_responsibilities(responsibilities)    
        return weights, means, covariances
    
    
class KmeansSklearnInit(Inits):
    def calculate_parameters(self):
        responsibilities = np.zeros((self.n_samples, self.n_components))
        label = (
            cluster.KMeans(
                n_clusters=self.n_components, n_init=1, random_state=self.random_state
            )
            .fit(self.X)
            .labels_
        )
        responsibilities[np.arange(self.n_samples), label] = 1
        weights, means, covariances = self.calculate_parameters_from_responsibilities(responsibilities)    
        return weights, means, covariances
    
    
_init_type_to_class_map = {'random': RandomInit, 'kmeans': KmeansInit, 'randomized_kmeans': RandomizedKmeansInit,
                           'minmax': MinMaxInit, 'random_sklearn': RandomSklearnInit, 'kmeans_sklearn': KmeansSklearnInit}

def initialize_parameters(X, random_state=None, n_components=1, init_type='random_sklearn', scale=None, reg_covar=0):
    """Initialize the Gaussian mixture model (GMM) parameters (weights, means, covariances and precisions).
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data based on which the GMM initialization is calculated.    
    random_state : int, default=None
        A random seed used for the method chosen to initialize the parameters.
    n_components : int, default=1
        Number of GMM components to initialize.
    init_type : {'random', 'minmax', 'kmeans', 'randomized_kmeans', 'random_sklearn', 'kmeans_sklearn'}, default='random_sklearn'
        The method used to initialize the weights, the means, the covariances and the precisions.
        Must be one of:
            'random': weights are set uniformly, covariances are proportional to identity (with prefactor s^2). 
            For each mean, a data sample is selected at random, and a multivariate Gaussian with variance s^2 offset is added.
            'minmax': same as above, but means are distributed randomly over the range that is covered by data.
            'kmeans': k-means clustering run as in Algorithm 1 from Bloemer & Bujna (arXiv:1312.5946), 
            as implemented by Melchior & Goulding (arXiv:1611.05806).
            'randomized_kmeans': same as k-means, but then the means are pushed around for increased likelihood 
            of avoiding local minima. The 'scale' is used to quantify the mean push, and can be specified by the user (defaults to 1).
            'random_sklearn': responsibilities are initialized randomly, i.e. every point is randomly assigned to a component.
            Weights, means and covariances are then calculated by performing an M-step on these responsibilities:
            this means that weights are calculated as the average probability that a sample belongs to each components, while
            means and covariance matrices are the weighted mean and covariance of the samples.
            'kmeans_sklearn': responsibilities are initialized using k-means, as implemented by sklearn. 
            The rest is as in 'random_sklearn'.
            Note we are currently not including the sklearn initializations available from v1.1.
    scale : float, default=None
        If set, sets component variances in the 'random' and 'minmax' cases. 
        If scale is not given, it will be set such that the volume of all components completely fills the space covered by data.
    reg_covar : float, default=0
        Constant regularisation term added to the diagonal of each covariance matrix, to avoid singular matrices.
        In GMM-MI, the regularisation term is added during training only, and not in the initializations.
        
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
    if init_type in ['random', 'kmeans', 'randomized_kmeans', 'minmax', 'random_sklearn', 'kmeans_sklearn']:
        InitClass = _init_type_to_class_map[init_type]
        init_class = InitClass(X=X, random_state=random_state, n_components=n_components, 
                               init_type=init_type, scale=scale, reg_covar=reg_covar)
        weights, means, covariances = init_class.calculate_parameters()
        covariances = init_class.add_regularization(covariances=covariances)
    else:
        raise ValueError(f"Initialization type not known. It should be one of "
                         f"'random', 'kmeans', 'randomized_kmeans', 'minmax', 'random_sklearn', 'kmeans_sklearn'; found '{init_type}'") 

    # all matrices can be inverted at once
    precisions = np.linalg.inv(covariances)
    return weights, means, covariances, precisions
