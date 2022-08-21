# Notebooks 

In this folder we share notebooks with example use cases of GMM-MI, as well as to reproduce all results from the paper. Results are usually stored in [`results`](https://github.com/dpiras/MI_estimation/tree/main/notebooks/results), and figures in (you guess it) [`figures`](https://github.com/dpiras/MI_estimation/tree/main/notebooks/figures). Configurations files to share the parameters across different experiments can be found in `*.yml` files. The folder [`MI_synthetic_datasets`](https://github.com/dpiras/MI_estimation/tree/main/notebooks/MI_synthetic_datasets) contains the true value of MI for the synthetic datasets found in [`gmm_mi/data`](https://github.com/dpiras/MI_estimation/tree/main/gmm_mi/data).

## List of available notebooks

- [0_calculate_MI_synthetic_datasets](https://github.com/dpiras/MI_estimation/blob/main/notebooks/0_calculate_MI_synthetic_datasets.ipynb): calculate MI for all the synthetic datasets found in `gmm_mi/data`.
- [1_MI_D3_significance](https://github.com/dpiras/MI_estimation/blob/main/notebooks/1_MI_D3_significance.ipynb): residual analysis for a dataset with 3 components, D3. Left panel of Fig. 2 in the paper.
- [2_MI_D3p_significance](https://github.com/dpiras/MI_estimation/blob/main/notebooks/2_MI_D3p_significance.ipynb): residual analysis for another dataset with 3 components, D3', where the components are more separated.
- [3_MI_D5_significance](https://github.com/dpiras/MI_estimation/blob/main/notebooks/3_MI_D5_significance.ipynb): residual analysis for a dataset with 5 components, D5. Middle panel of Fig. 2 in the paper.
- [4_MI_D5_significance](https://github.com/dpiras/MI_estimation/blob/main/notebooks/4_MI_D5p_significance.ipynb): residual analysis for a dataset with 5 components, D5'. This is identical to D5, but one component is shifted to the right. Right panel of Fig. 2 in the paper.
- [5_Gaussian_comparison](https://github.com/dpiras/MI_estimation/blob/main/notebooks/5_Gaussian_comparison.ipynb): comparison between GMM-MI, KSG and MINE estimators on a Gaussian dataset with varying level of correlation. Top panel of Fig. 1 in the paper.

