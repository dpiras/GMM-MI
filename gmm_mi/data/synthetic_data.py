import numpy as np
from gmm_mi.gmm import GMMWithMI as GMM
random_state = 13

# We define a few different synthetic GMM models, with different components,
# which are used to test GMM-MI. The random state is fixed.
# The suffix 'p' indicates models with different means only.

D1 = GMM(n_components = 1,
            weights_init = np.array([1.0]),
            means_init = np.array([[-1, 1]]),
            covariances_init = np.array([[[0.5, 0.2], [0.2, 0.5]]]),
            random_state = random_state)
 
D3 = GMM(n_components = 3,
            weights_init = np.array([0.3, 0.45, 0.25]),
            means_init = np.array([[-1, 1], [0, 2], [-1.5, 2]]),
            covariances_init = np.array([[[1, 0], [0, 0.1]], 
                                         [[0.5, 0.2], [0.2, 0.5]], 
                                         [[0.1, 0.05], [0.05, 0.1]]]),
            random_state = random_state)
 
D3p = GMM(n_components = 3,
            weights_init = np.array([0.3, 0.45, 0.25]),
            means_init = np.array([[-3, 3], [2, 4], [-2.5, 5]]),
            covariances_init = np.array([[[1, 0], [0, 0.1]], 
                                         [[0.5, 0.2], [0.2, 0.5]], 
                                         [[0.1, 0.05], [0.05, 0.1]]]),
            random_state = random_state)

D3_rhoneq0 = GMM(n_components = 3,
            weights_init = np.array([0.3, 0.45, 0.25]),
            means_init = np.array([[-1, 1], [0, 2], [-1.5, 2]]),
            covariances_init = np.array([[[1, 0.1], [0.1, 0.1]], 
                                         [[0.5, 0.2], [0.2, 0.5]], 
                                         [[0.1, 0.05], [0.05, 0.1]]]),
            random_state = random_state)

D5 = GMM(n_components = 5,
            weights_init = np.array([0.2, 0.35, 0.15, 0.12, 0.18]),
            means_init = np.array([[-1, 1], [0, 2], [-1.5, 2], [2, 1], [-0.25, 0]]),
            covariances_init = np.array([[[1, 0], [0, 0.1]], 
                                         [[0.5, 0.2], [0.2, 0.5]], 
                                         [[0.1, 0.05], [0.05, 0.1]], 
                                         [[0.5, -0.1], [-0.1, 0.9]], 
                                         [[0.2, -0.05], [-0.05, 0.1]]]),
            random_state = random_state)

D5p = GMM(n_components = 5,
            weights_init = np.array([0.2, 0.35, 0.15, 0.12, 0.18]),
            means_init = np.array([[-1, 1], [0, 2], [-1.5, 2], [3, 1], [-0.25, 0]]),
            covariances_init = np.array([[[1, 0], [0, 0.1]], 
                                         [[0.5, 0.2], [0.2, 0.5]], 
                                         [[0.1, 0.05], [0.05, 0.1]], 
                                         [[0.5, -0.1], [-0.1, 0.9]], 
                                         [[0.2, -0.05], [-0.05, 0.1]]]),
            random_state = random_state)
