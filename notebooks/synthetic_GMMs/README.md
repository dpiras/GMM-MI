# Notebooks

In this folder we share notebooks with example use cases of GMM-MI on synthetic GMMs. Pre-computed results are stored in [`results`](https://github.com/dpiras/MI_estimation/tree/main/notebooks/synthetic_GMMs/results), and figures in (you guess it) [`figures`](https://github.com/dpiras/MI_estimation/tree/main/notebooks/synthetic_GMMs/figures). The folder [`MI_synthetic_datasets`](https://github.com/dpiras/MI_estimation/tree/main/notebooks/synthetic_GMMs/MI_synthetic_datasets) contains the true value of MI for the synthetic datasets found in [`gmm_mi/data`](https://github.com/dpiras/MI_estimation/tree/main/gmm_mi/data). These results are not shown in the paper.

## List of available notebooks

- [0_calculate_MI_synthetic_datasets](https://github.com/dpiras/MI_estimation/blob/main/notebooks/synthetic_GMMs/0_calculate_MI_synthetic_datasets.ipynb): calculate MI for all the synthetic datasets found in `gmm_mi/data`.
- [1_MI_D3_significance](https://github.com/dpiras/MI_estimation/blob/main/notebooks/synthetic_GMMs/1_MI_D3_significance.ipynb): residual analysis for a dataset with 3 components, D3.
- [2_MI_D3p_significance](https://github.com/dpiras/MI_estimation/blob/main/notebooks/synthetic_GMMs/2_MI_D3p_significance.ipynb): residual analysis for another dataset with 3 components, D3', where the components are more separated.
- [3_MI_D5_significance](https://github.com/dpiras/MI_estimation/blob/main/notebooks/synthetic_GMMs/3_MI_D5_significance.ipynb): residual analysis for a dataset with 5 components, D5.
- [4_MI_D5_significance](https://github.com/dpiras/MI_estimation/blob/main/notebooks/synthetic_GMMs/4_MI_D5p_significance.ipynb): residual analysis for a dataset with 5 components, D5'. This is identical to D5, but one component is shifted to the right.
- [5_MI_D30_significance](https://github.com/dpiras/MI_estimation/blob/main/notebooks/synthetic_GMMs/5_MI_D30_significance.ipynb): residual analysis for another dataset with 3 components, D30 or D3_rhoneq0, where each component has a correlation coefficient different than 0.

