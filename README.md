# GMM-MI 

<p align="center">
  <img src="https://user-images.githubusercontent.com/25639122/195098930-93a9865b-a0c7-4792-9474-dc0d1056e358.png?raw=true" alt="GMM-MI_logo"/>
</p>

Welcome to GMM-MI (pronounced ``Jimmie``)! This package allows you to calculate mutual information (MI) with its associated uncertainty, combining Gaussian mixture models (GMMs) and bootstrap. GMM-MI is accurate, computationally efficient and fully in python; you can read more about GMM-MI [in our paper](https://iopscience.iop.org/article/10.1088/2632-2153/acc444), published in Machine Learning: Science and Technology. Please [cite it](#citation) if you use it in your work! Check out also the [poster accepted at the Machine Learning and the Physical Sciences workshop at NeurIPS 2022](https://neurips.cc/media/PosterPDFs/NeurIPS%202022/56922.png), and the accompanying [video](https://user-images.githubusercontent.com/25639122/201700436-3c3f6216-1925-4a09-9b04-419a64bfda15.mp4).

## Installation

To install GMM-MI, follow these steps:
1. (optional) `conda create -n gmm_mi python=3.9 jupyter` (create a custom `conda` environment with python 3.9) 
2. (optional) `conda activate gmm_mi` (activate it)
3. Install GMM-MI:

        pip install gmm-mi
        python3 -c 'from gmm_mi.mi import EstimateMI'

   or alternatively, clone the repository and install it:

        git clone https://github.com/dpiras/GMM-MI.git
        cd GMM-MI
        pip install . 
        pytest 

The latter option will also give you access to Jupyter notebooks to get started with GMM-MI. Note that all experiments were run with `python==3.9`, and that GMM-MI requires at least `python>=3.8`. 

## Usage

To use GMM-MI, you simply need to import the class `EstimateMI`, choose the hyperparameters and fit your data. You can find an example application of GMM-MI in the next section, and a more complete walkthrough, with common scenarios and possible pitfalls, in this [notebook](https://github.com/dpiras/GMM-MI/blob/main/notebooks/walkthrough_and_pitfalls.ipynb). A description of the hyperparameters that you can play with can be found [here](https://github.com/dpiras/GMM-MI/blob/main/gmm_mi/mi.py#L10), and we discuss a few of them [below](#hyperparameter-description).

## Example

Once you installed GMM-MI, calculating the distribution of mutual information on your data is as easy as:

    import numpy as np
    from gmm_mi.mi import EstimateMI

    # create simple bivariate Gaussian data
    mean, cov = np.array([0, 0]), np.array([[1, 0.6], [0.6, 1]])
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal(mean, cov, 200) # has shape (200, 2)
    # calculate MI
    mi_estimator = EstimateMI()
    MI_mean, MI_std = mi_estimator.fit_estimate(X)

This yields (0.21 &pm; 0.04) nat, well in agreement with the theoretical value of 0.22 nat. There are many things that you can do: for example, you can also pass two 1D arrays instead of a single 2D array, and even calculate the KL divergence between the marginals (as shown in the walkthrough notebook). If you want to visualize the fitted model over your input data, you can run:
    
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

You can also draw contour plots for the input data and samples obtained from the fitted model. For example (smoothness of the contour plot heavily depends on the number of samples available):

    fig = mi_estimator.plot_fitted_contours(parameters=['X1', 'X2'], 
                                        shade_alpha=0.4, linewidths=2, 
                                        legend_kwargs = {'loc': 'lower right'},
                                        kde=True, # smooths contours; set this to False to accelerate plotting
                                        )
    fig.set_size_inches(7, 7)

To choose the GMM-MI hyperparameters, we provide three classes: `GMMFitParamHolder`, `SelectComponentsParamHolder`, and `MIDistParamHolder`. An example is as follows:

    from gmm_mi.param_holders import GMMFitParamHolder, SelectComponentsParamHolder, MIDistParamHolder

    # parameters for every GMM fit that is being run
    gmm_fit_params = GMMFitParamHolder(threshold_fit=1e-5, reg_covar=1e-15)
    # parameters to choose the number of components
    select_components_params = SelectComponentsParamHolder(n_inits=3, n_folds=2)
    # parameters for MI distribution estimation
    mi_dist_params = MIDistParamHolder(n_bootstrap=50, MC_samples=1e5)

    mi_estimator = EstimateMI(gmm_fit_params=gmm_fit_params,
                              select_components_params=select_components_params)
    mi_estimator.fit(X)
    MI_mean, MI_std = mi_estimator.estimate(mi_dist_params=mi_dist_params)

This is equivalent to the first example, and yields (0.21 &pm; 0.04) nat. More example notebooks, including conditional mutual information and all results from the paper, are available in [`notebooks`](https://github.com/dpiras/GMM-MI/blob/main/notebooks).

## Hyperparameter description
Here we report the most important hyperparameters that are used in GMM-MI.

    (controlled by GMMFitParamHolder, passed as gmm_fit_params)
    threshold_fit : float, default=1e-5
        The log-likelihood threshold on each GMM fit used to choose when to stop training. Smaller
        values will improve the fit quality and reduce the chances of stopping at a local optimum,
        while making the code considerably slower. This is equivalent to `tol` in sklearn GMMs.
        Note this parameter can be degenerate with `threshold_components`, and the two should be set
        together to reach a good density estimate of the data.
    reg_covar : float, default=1e-15
        The constant term added to the diagonal of the covariance matrices to avoid singularities.
        Smaller values will increase the chances of singular matrices, but will have a smaller
        impact on the final MI estimates.
    init_type : {'random', 'minmax', 'kmeans', 'randomized_kmeans', 'random_sklearn', 
                 'kmeans_sklearn'}, default='random_sklearn'
        The method used to initialize the weights, the means, the covariances and the precisions
        in each fit during cross-validation. See utils.initializations for more details.
    scale : float, default=None
        The scale used for 'random', 'minmax' and 'randomized_kmeans' initializations. 
        This hyperparameter is not used in all other cases, but it is useful if you roughly know 
        in advance the scale of your data, and can accelerate convergence.
        See utils.initializations for more details.

    (controlled by ChooseComponentParamHolder, passed as select_component_params)
    n_inits : int, default=3
        Number of initializations used to find the best initialization parameters. Higher
        values will decrease the chances of stopping at a local optimum, while making the
        code slower.
    n_folds : int, default=2
        Number of folds in the cross-validation (CV) performed to find the best initialization
        parameters. As in every CV procedure, there is no best value. A good value, though,
        should ensure each fold has enough samples to be representative of your training set.
    threshold_components : float, default=1e-5
        The metric threshold to decide when to stop adding GMM components. In other words, GMM-MI
        stops adding components either when the metric gets worse, or when the improvement in the
        metric value is less than this threshold. Smaller values ensure that enough components are
        considered and that the data distribution is correctly captured, while taking longer to converge.
        Note this parameter can be degenerate with `threshold_fit`, and the two should be set 
        together to reach a good density estimate of the data.
    patience : int, default=1 
        Number of extra components to "wait" until convergence is declared. Must be at least 1.
        Same concept as patience when training a neural network. Higher value will fit models
        with higher numbers of GMM components, while taking longer to converge.
   
    (controlled by MIDistParamHolder, passed as mi_dist_params to the `estimate` method) 
    n_bootstrap : int, default=50 
        Number of bootstrap realisations to consider to obtain the MI uncertainty.
        Higher values will return a better estimate of the MI uncertainty, and
        will make the MI distribution more Gaussian-like, but will take longer.
        If less than 1, do not perform bootstrap and actually just do a single 
        fit on the entire dataset; there will be no MI uncertainty in this case.
    MC_samples : int, default=1e5
        Number of MC samples to use to estimate the MI integral. Only used if MI_method == 'MC'.
        Higher values will return less noisy estimates of MI, but will take longer.

## Contributing and contacts

Feel free to [fork](https://github.com/dpiras/GMM-MI/fork) this repository to work on it; otherwise, please [raise an issue](https://github.com/dpiras/GMM-MI/issues) or contact [Davide Piras](mailto:dr.davide.piras@gmail.com).

## Citation
If you use GMM-MI, please cite the corresponding paper:

     @article{Piras23, 
          author = {Davide Piras and Hiranya V Peiris and Andrew Pontzen and 
                    Luisa Lucie-Smith and Ningyuan Guo and Brian Nord},
          title = {A robust estimator of mutual information for deep learning interpretability},
          journal = {Machine Learning: Science and Technology},
          doi = {10.1088/2632-2153/acc444},
          url = {https://dx.doi.org/10.1088/2632-2153/acc444},
          year = {2023},
          month = {apr},
          publisher = {IOP Publishing},
          volume = {4},
          number = {2},
          pages = {025006}
    }

## License

GMM-MI is released under the GPL-3 license - see [LICENSE](https://github.com/dpiras/GMM-MI/blob/main/LICENSE.txt)-, subject to 
the non-commercial use condition - see [LICENSE_EXT](https://github.com/dpiras/GMM-MI/blob/main/LICENSE_EXT.txt).

     GMM-MI
     Copyright (C) 2022 Davide Piras & contributors

     This program is released under the GPL-3 license (see LICENSE.txt), 
     subject to a non-commercial use condition (see LICENSE_EXT.txt).

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
