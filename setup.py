#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: 3-clause BSD
import os
from setuptools import setup, find_packages

__version__ = "0.0.0"
NAME = 'BraiNets'
AUTHOR = "BraiNets"
MAINTAINER = "Andrea Brovelli"
EMAIL = 'andrea.brovelli@univ-amu.fr'
KEYWORDS = "multi-modal fmri meg seeg causality gaussian copula"
DESCRIPTION = "Multi-modal causality"
URL = 'https://github.com/brainets/brainets'
DOWNLOAD_URL = ("https://github.com/brainets/brainets/archive/v" +
                __version__ + ".tar.gz")
# Data path :
PACKAGE_DATA = {}


def read(fname):
    """Read README and LICENSE."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name=NAME,
    version=__version__,
    packages=find_packages(),
    package_dir={'brainets': 'brainets'},
    package_data=PACKAGE_DATA,
    include_package_data=True,
    description=DESCRIPTION,
    long_description=read('README.rst'),
    platforms='any',
    setup_requires=['numpy'],
    install_requires=[
        "numpy",
        "scipy",
        "mne",
        "pandas",
        "xarray",
        "joblib",
        "matplotlib"
    ],
    dependency_links=[],
    author=AUTHOR,
    maintainer=MAINTAINER,
    author_email=EMAIL,
    url=URL,
    download_url=DOWNLOAD_URL,
    license=read('LICENSE'),
    keywords=KEYWORDS,
    classifiers=["Development Status :: 3 - Alpha",
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: Education',
                 'Intended Audience :: Developers',
                 'Topic :: Scientific/Engineering :: Visualization',
                 "Programming Language :: Python :: 3.5",
                 "Programming Language :: Python :: 3.6",
                 "Programming Language :: Python :: 3.7"
                 ])
