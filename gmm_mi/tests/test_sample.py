import numpy as np
from gmm_mi.mi import EstimateMI
from gmm_mi.param_holders import MIDistParamHolder

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


def test_KL_analytic():
    nu_a, std_a = 7, 4
    nu_b, std_b = -3, 5
    a = np.random.normal(nu_a, std_a, 10000)
    b = np.random.normal(nu_b, std_b, 10000)
    analytic_kl = np.log(std_b) - np.log(std_a) - 0.5 * (1 - ((std_a ** 2 + (nu_a - nu_b) ** 2) / std_b ** 2))
    mi_mean, mi_std = EstimateMI().fit(np.column_stack((a, b)), kl=True)
    assert (analytic_kl >= (mi_mean - 2*mi_std)) and (analytic_kl <= (mi_mean + 2*mi_std))
    mi_mean1, mi_std1 = EstimateMI(mi_dist_params=MIDistParamHolder(MI_method='quad')).fit(np.column_stack((a, b)), kl=True)
    assert (analytic_kl >= (mi_mean1 - 2 * mi_std1)) and (analytic_kl <= (mi_mean1 + 2 * mi_std1))

