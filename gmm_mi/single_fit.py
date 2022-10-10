import numpy as np
from gmm_mi.gmm import GMMWithMI as GMM


def single_fit(X, n_components, reg_covar=1e-15, threshold_fit=1e-5, max_iter=10000, 
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
    threshold_fit : float, default=1e-5
        The log-likelihood threshold on each GMM fit used to choose when to stop training.
        Smaller values will improve the fit quality and reduce the chances of stopping at a local optimum,
        while making the code considerably slower. This is equivalent to `tol` in sklearn GMMs.       
    max_iter : int, default=10000
        The maximum number of iterations in each GMM fit. We aim to stop only based on `threshold_fit`, 
        so it is set to a high value. A warning is raised if this threshold is reached; in that case, 
        simply increase this value, or check that you really need such a small `threshold_fit` value.
    random_state : int, default=None
        Random seed used to initialize the GMM model. 
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
                threshold_fit=threshold_fit, max_iter=max_iter, 
                random_state=random_state, weights_init=w_init, 
                means_init=m_init, precisions_init=p_init).fit(X, val_set=val_set)
    return gmm