# GMM-MI 
Welcome to GMM-MI! This documentation is a work in progress.

Current missing features include:
- full documentation and testing of the main module gmm_mi.py
- more test notebooks, including all results from the paper
- the installation instructions (including an example calculation of MI).

## Current state

As of July 30th 2022, we are observing a small MI bias when applying our estimator. Please check [this jupyter notebook](https://github.com/dpiras/MI_estimation/blob/main/notebooks/bias_MI_D3p.ipynb) for a minimum working example, and do not hesitate to let me know if you have any problems with it.

## Installation

To install GMM-MI, we currently recommend the following steps:
1. Create custom `conda` environment: `conda create -n "gmm_mi" python=3.7`
2. Activate it: `conda activate gmm_mi`
3. Clone repository (with https you need to insert your GH credentials): `git clone https://github.com/dpiras/MI_estimation.git`
4. Move into the cloned folder: `cd MI_estimation`
5. Install `gmm_mi` and all its dependencies: `python setup.py install`

We will make the package `pip` installable once we make the repository public.

## Example

Once you installed GMM-MI, calculating the distribution of mutual information on your data is as easy as

    TBC

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
