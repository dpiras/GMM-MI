import numpy as np


def nonlinear_transformation(x, transformation='identity'):
    """Transforms marginal of 2D array according to the required function.
    
    Parameters
    ----------
    x : array-like of shape (n_samples, 2)
        Data to transform.
    transformation : string, default='identity'
        One of 'identity', 'square', 'cube' or 'log'. 
        Transform data according to this function. 
        If not specified, or different than the above, return the input array
    
    Returns
    ----------
    x : array-like of shape (n_samples, 2)
        The transformed array.
    """    
    if transformation == 'identity':
        x[:, 1] = x[:, 1]
    elif transformation == 'square':
        x[:, 1] = x[:, 1] + 0.1 * x[:, 1]**2 
    elif transformation == 'cube':
        x[:, 1] = x[:, 1] + 0.5 * x[:, 1]**3 
    elif transformation == 'log':
        x[:, 1] = np.log(x[:, 1]+5.0)      
    return x