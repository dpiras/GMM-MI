from setuptools import setup, find_packages
import shutil

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

PACKAGENAME = 'gmm_mi'

setup(
    name='gmm_mi',
    version="0.4.1",
    author='Davide Piras',
    author_email='dr.davide.piras@gmail.com',
    description='Estimate mutual information distribution with Gaussian mixture models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/dpiras/GMM-MI',
    license='GNU General Public License v3.0 (GPLv3)',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=['pytest',
                      'tqdm',
                      'matplotlib>=3.1.2',
                      'numpy>=1.17.4',
                      'scikit-learn>=1.0.2',
                      'scipy>=1.7.1']
                      )

