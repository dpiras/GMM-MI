import numpy as np
from gmm_mi.mi import EstimateMI
from gmm_mi.param_holders import MIDistParamHolder

def test_gaussian():
    true_mean, true_std = 0.2140917, 0.0429001
    # create simple bivariate Gaussian data
    mean, cov = np.array([0, 0]), np.array([[1, 0.6], [0.6, 1]])
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal(mean, cov, 200) # has shape (200, 2)
    # calculate MI
    mi_estimator = EstimateMI()
    MI_mean, MI_std = mi_estimator.fit_estimate(X)

    assert np.allclose(MI_mean, true_mean)
    assert np.allclose(MI_std, true_std)

    # check nothing changes if we pass two 1D arrays instead of a single 2D array
    mi_estimator = EstimateMI()
    MI_mean, MI_std = mi_estimator.fit_estimate(X[:, 0], X[:, 1])

    assert np.allclose(MI_mean, true_mean)
    assert np.allclose(MI_std, true_std)

    # also check these do not fail
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # X is the array with the input data
    ax.scatter(X[:, 0], X[:, 1], label='Input data')
    # the extra arguments can be changed
    ax = mi_estimator.plot_fitted_model(ax=ax, color='salmon', alpha=0.8, linewidth=4)
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax.set_xlabel('X', fontsize=30)
    ax.set_ylabel('Y', fontsize=30)
    ax.legend(fontsize=25, frameon=False)


def test_KL_analytic_direct():
    # test KL is in agreement within some sigmas for Gaussian data
    nu_a, std_a = 7, 4
    nu_b, std_b = -3, 5
    sigmas = 3
    N = 10000
    a = np.random.normal(nu_a, std_a, N)
    b = np.random.normal(nu_b, std_b, N)
    analytic_kl = np.log(std_b) - np.log(std_a) - 0.5 * (1 - ((std_a ** 2 + (nu_a - nu_b) ** 2) / std_b ** 2))
    mi_estimator = EstimateMI()
    _, _ = mi_estimator.fit_estimate(a, b, include_kl=True)
    kl_mean, kl_std = mi_estimator.KL_mean, mi_estimator.KL_std
    assert (analytic_kl >= (kl_mean - sigmas * kl_std)) and (analytic_kl <= (kl_mean + sigmas * kl_std))


def test_KL_analytic_inverse():
    # test KL is in agreement within some sigmas for Gaussian data
    nu_a, std_a = 7, 4
    nu_b, std_b = -3, 5
    sigmas = 3
    N = 10000
    a = np.random.normal(nu_a, std_a, N)
    b = np.random.normal(nu_b, std_b, N)
    analytic_kl = np.log(std_a) - np.log(std_b) - 0.5 * (1 - ((std_b ** 2 + (nu_b - nu_a) ** 2) / std_a ** 2))
    mi_estimator = EstimateMI()
    _, _ = mi_estimator.fit_estimate(a, b, include_kl=True, kl_order='reverse')
    kl_mean, kl_std = mi_estimator.KL_mean, mi_estimator.KL_std
    assert (analytic_kl >= (kl_mean - sigmas * kl_std)) and (analytic_kl <= (kl_mean + sigmas * kl_std))

