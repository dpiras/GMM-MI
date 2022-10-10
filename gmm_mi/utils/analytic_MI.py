import numpy as np
from scipy.special import psi 


def calculate_MI_D1_analytical(covariance_matrix):
    """Calculate the mutual information (MI) for a single bivariate Gaussian using the analytical formula.
    In this simple setting, MI only depends on the correlation coefficient between the two variables.
    
    Parameters
    ----------
    covariance_matrix : array-like of shape (2, 2)
        Covariance matrix of the bivariate Gaussian.
    
    Returns
    ----------
    MI : float
        The value of mutual information in nat.
    """
    assert covariance_matrix.shape==(2, 2), "Covariance matrix must be of shape (2, 2)."
    MI = -0.5*np.log(1-(covariance_matrix[0, 1] / (np.sqrt(covariance_matrix[0, 0]*covariance_matrix[1, 1])))**2)
    return MI


def calculate_MI_gammaexp_analytical(alpha):
    """Calculate the mutual information (MI) for a gamma-exponential distribution, with free parameter \alpha.
    
    Parameters
    ----------
    alpha : float
        Free parameter of the distribution.
    
    Returns
    ----------
    MI : float
        The value of mutual information in nat.
    """
    # psi is the digamma function
    return psi(alpha+1) - np.log(alpha)


def calculate_MI_weinman_analytical(alpha):
    """Calculate the mutual information (MI) for an ordered Weinman exponential distribution, with free parameter \alpha.
    
    Parameters
    ----------
    alpha : float
        Free parameter of the distribution.
    
    Returns
    ----------
    MI : float
        The value of mutual information in nat.
    """
    # psi is the digamma function
    if 0 < alpha < 0.5:
        return -np.log(2*alpha / (1-2*alpha)) + psi( 1 / (1-2*alpha)) -psi(1)
    elif alpha == 0.5:
        return -psi(1)
    else:
        return np.log( (2*alpha-1)  / (2*alpha)) + psi( (2*alpha) / (2*alpha-1)) -psi(1)
