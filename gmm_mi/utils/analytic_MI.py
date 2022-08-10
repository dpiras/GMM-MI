import numpy as np


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
        The value of mutual information in nats.
    """
    assert covariance_matrix.shape==(2, 2), "Covariance matrix must be of shape (2, 2)."
    MI = -0.5*np.log(1-(covariance_matrix[0, 1] / (np.sqrt(covariance_matrix[0, 0]*covariance_matrix[1, 1])))**2)
    return MI
