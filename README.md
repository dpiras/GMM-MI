# GMM-MI 

Welcome to GMM-MI (pronounced ``Jimmie``)! This package allows you to calculate mutual information (MI) with its associated uncertainty, combining Gaussian mixture models (GMMs) and bootstrap. You can read more about GMM-MI [in our paper](https://www.overleaf.com/project/62920145c884448df7e9745c) (the link will be to the actual paper once submitted).

Current missing features include:
- more test notebooks, including all results from the paper

## Current state

As of August 11th 2022, we have collected most results and only need to make them available in the `notebooks` folder. We are observing a small MI bias when applying our estimator; please check [this jupyter notebook](https://github.com/dpiras/MI_estimation/blob/main/notebooks/1_MI_D3p_significance.ipynb) for a simple example. This will be addressed soon, as we put together all the results that will go in the paper.

## Installation

To install GMM-MI, we currently recommend the following steps:
1. `conda create -n "gmm_mi" python=3.7` (create custom `conda` environment) 
2. `conda activate gmm_mi` (activate it)
3. `git clone https://github.com/dpiras/MI_estimation.git` (clone repository; with `https` you need to insert your GH credentials)
4. `cd MI_estimation` (move into cloned folder)
5. `python setup.py install` (install `gmm_mi` and all its dependencies)

We will make the package `pip` installable once we make the repository public.

## Example

Once you installed GMM-MI, calculating the distribution of mutual information on your data is as easy as:

    import numpy as np
    from gmm_mi.gmm_mi import GMM_MI
    # create simple bivariate Gaussian data
    mean, cov = np.array([0, 0]), np.array([[1, 0.6], [0.6, 1]])
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal(mean, cov, 200) # has shape (200, 2)
    # calculate MI
    MI_mean, MI_std = GMM_MI(X)

This yields (0.21 &pm; 0.04) nats, well in agreement with the theoretical value of 0.22 nats.
A description of the hyperparameters that you can play with can be found [here](https://github.com/dpiras/MI_estimation/blob/main/gmm_mi/gmm_mi.py#L428).
More example notebooks, including all results from the paper, are available in [`notebooks`](https://github.com/dpiras/MI_estimation/blob/main/notebooks).

## Contributing and contacts
Feel free to [fork](https://github.com/dpiras/MI_estimation/fork) this repository to work on it; otherwise, please [raise an issue](https://github.com/dpiras/MI_estimation/issues) or contact [Davide Piras](mailto:d.piras@ucl.ac.uk).

## Citation
If you use GMM-MI, please cite the corresponding paper:

     @article{TBC, 
        author = {TBC},
         title = {TBC},
       journal = {TBC},
        eprint = {TBC},
          year = {TBC}
     }

## License

GMM-MI is released under the GPL-3 license - see [LICENSE](https://github.com/dpiras/MI_estimation/blob/main/LICENSE.txt)-, subject to 
the non-commercial use condition - see [LICENSE_EXT](https://github.com/dpiras/MI_estimation/blob/main/LICENSE_EXT.txt).

     GMM-MI
     Copyright (C) 2022 Author names & contributors

     This program is released under the GPL-3 license (see LICENSE.txt), 
     subject to a non-commercial use condition (see LICENSE_EXT.txt).

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
