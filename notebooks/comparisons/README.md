# Notebooks 

In this folder we share notebooks with comparisons of GMM-MI with other estimators of MI. Results are usually stored in [`results`](https://github.com/dpiras/MI_estimation/tree/main/notebooks/comparisons/results), and figures in (did you guess it?) [`figures`](https://github.com/dpiras/MI_estimation/tree/main/notebooks/comparisons/figures). The folder [`mine_pytorch`](https://github.com/dpiras/MI_estimation/tree/main/notebooks/comparisons/mine-pytorch) contains a version of the MINE estimator implementation available [at this repository](https://github.com/gtegner/mine-pytorch).

## List of available notebooks

- [1_Gaussian_comparison](https://github.com/dpiras/MI_estimation/blob/main/notebooks/comparisons/1_Gaussian_comparison.ipynb): comparison between GMM-MI, KSG and MINE estimators on a Gaussian distribution with varying level of correlation. Top panel of Fig. 1 in the paper.
- [2_gamma_exp_comparison](https://github.com/dpiras/MI_estimation/blob/main/notebooks/comparisons/2_gamma_exp_comparison.ipynb): comparison between GMM-MI, KSG and MINE estimators on a gamma-exponential distribution with varying $\alpha$. Middle panel of Fig. 1 in the paper.
- [3_weinman_exp_comparison](https://github.com/dpiras/MI_estimation/blob/main/notebooks/comparisons/3_weinman_exp_comparison.ipynb): comparison between GMM-MI, KSG and MINE estimators on a Weinman exponential distribution with varying $\alpha$. Bottom panel of Fig. 1 in the paper.
- [4_bootstrap_comparison](https://github.com/dpiras/MI_estimation/blob/main/notebooks/comparisons/4_bootstrap_comparison.ipynb): comparison between GMM-MI, KSG and MINE estimators when performing bootstrap to obtain the full MI distribution on a bivariate Gaussian dataset. Fig. 3 in the paper.
- [5_metric_comparison](https://github.com/dpiras/MI_estimation/blob/main/notebooks/comparisons/5_metric_comparison.ipynb): comparison of validation log-likelihood, Akaike information criterion (AIC) and Bayesian information criterion (BIC) for a particular pair of latent-radial bin in the halo data. Fig. 7 in the paper.

