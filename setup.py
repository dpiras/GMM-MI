from setuptools import setup, find_packages

PACKAGENAME = 'gmm_mi'

setup(
    name='gmm_mi',
    version="0.1.0",
    author='Davide Piras',
    author_email='d.piras@ucl.ac.uk',
    description='Estimate mutual information distribution with Gaussian mixture models',
    url='https://github.com/dpiras/MI_estimation',
    license='GNU General Public License v3.0 (GPLv3)',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPLv3 License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=['matplotlib==3.1.2',
                      'numpy==1.17.4',
                      'pandas==1.3.4',
                      'scikit-learn==1.0.2',
                      'scipy==1.7.1']
                      )
