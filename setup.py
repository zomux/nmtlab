#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nmtlab
import sys, os

if sys.version_info[:2] < (3, 0):
    raise Exception('This version needs Python 3 or later. ')

from setuptools import setup, find_packages

requirements = ["numpy", "torch", "sacrebleu"]

setup(
    name='nmtlab',
    version=nmtlab.__version__,
    description='A simple framework for neural machine translation based on PyTorch',

    author='Raphael Shu',
    author_email='raphael@uaca.com',

    url='https://github.com/zomux/nmtlab',
    download_url='http://pypi.python.org/pypi/nmtlab',

    keywords=' Deep learning '
        ' Neural network '
        ' Natural language processing '
        ' Machine Translation ',

    license='MIT',
    platforms='any',

    packages=list(filter(lambda s: "private" not in s, find_packages())),

    classifiers=[ # from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.1',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
    ],

    setup_requires = requirements,
    install_requires=requirements,

    extras_require={

    },

    include_package_data=True,
)