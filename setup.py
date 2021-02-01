import os
from setuptools import setup
from setuptools import find_packages

setup(
    name='mGPLVM',
    author='Ta-Chu Kao and Kris Jensen',
    version='0.0.1',
    description='Pytorch implementation of mGPLVM',
    license='MIT',
    install_requires=['numpy', 'torch>=0.4.1', 'scipy>=1.0.0', 'scikit-learn'],
    packages=find_packages())
