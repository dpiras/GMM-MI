import numpy as np
from gmm_mi.mi import EstimateMI

def test_simple():
    # create simple bivariate Gaussian data
    mean, cov = np.array([0, 0]), np.array([[1, 0.6], [0.6, 1]])
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal(mean, cov, 200) # has shape (200, 2)
    # calculate MI
    mi_estimator = EstimateMI()
    MI_mean, MI_std = mi_estimator.fit(X)

    assert np.allclose(MI_mean, 0.2140917)
    assert np.allclose(MI_std, 0.0429001)

    # also check these do not fail
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # X is the array with the input data
    ax.scatter(X[:, 0], X[:, 1], label='Input data')
    # the extra arguments can be changed
    ax = mi_estimator.plot_fitted_model(ax=ax, color='salmon', alpha=0.8, linewidth=4)
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax.set_xlabel('X1', fontsize=30)
    ax.set_ylabel('X2', fontsize=30)
    ax.legend(fontsize=25, frameon=False)   
