# GMM-MI 

Welcome to GMM-MI (pronounced ``Jimmie``)! This package allows you to calculate mutual information (MI) with its associated uncertainty, combining Gaussian mixture models (GMMs) and bootstrap. GMM-MI is computationally efficient and fully in python. You can read more about GMM-MI [in our paper](https://www.overleaf.com/project/62920145c884448df7e9745c) (the link will be to the actual paper once submitted). Please [cite it](#citation) if you use it in your work!

Current missing features include:
- more test notebooks, including all results from the paper

## Installation

To install GMM-MI, we currently recommend the following steps:
1. (optional) `conda create -n gmm_mi python=3.7 jupyter` (we recommend creating a custom `conda` environment) 
2. (optional) `conda activate gmm_mi` (activate it)
3. `git clone https://github.com/dpiras/MI_estimation.git` (clone repository; with `https` you need to insert your GH credentials)
4. `cd MI_estimation` (move into cloned folder)
5. `python setup.py install` (install `gmm_mi` and all its dependencies); alternatively, `pip install .` should also work.
6. `pytest` (to make sure the installation worked correctly)

We will make the package `pip` installable once we make the repository public, and update these instructions.

## Usage

To use GMM-MI, you simply need to import the class EstimateMI, choose the hyperparameters and fit it to your data. A description of the hyperparameters that you can play with can be found [here](https://github.com/dpiras/MI_estimation/blob/main/gmm_mi/mi.py#L6). You can find an example below.

## Example

Once you installed GMM-MI, calculating the distribution of mutual information on your data is as easy as:

    import numpy as np
    from gmm_mi.mi import EstimateMI
    # create simple bivariate Gaussian data
    mean, cov = np.array([0, 0]), np.array([[1, 0.6], [0.6, 1]])
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal(mean, cov, 200) # has shape (200, 2)
    # calculate MI
    MI_mean, MI_std = EstimateMI().fit(X)

This yields (0.21 &pm; 0.04) nats, well in agreement with the theoretical value of 0.22 nats.
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
     Copyright (C) 2022 Davide Piras & contributors

     This program is released under the GPL-3 license (see LICENSE.txt), 
     subject to a non-commercial use condition (see LICENSE_EXT.txt).

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
