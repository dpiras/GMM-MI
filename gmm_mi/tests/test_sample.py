import numpy as np
from gmm_mi.mi import EstimateMI

def test_simple():
    # create simple bivariate Gaussian data
    mean, cov = np.array([0, 0]), np.array([[1, 0.6], [0.6, 1]])
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal(mean, cov, 200) # has shape (200, 2)
    # calculate MI
    MI_mean, MI_std = EstimateMI().fit(X)

    assert np.allclose(MI_mean, 0.2140917)
    assert np.allclose(MI_std, 0.0429001)
