import os
from setuptools import setup
from setuptools import find_packages

setup(name='mGPLVM',
      author='Ta-Chu Kao and Kris Jensen',
      version='0.0.1',
      description='Pytorch implementation of mGPLVM and bGPFA',
      license='MIT',
      install_requires=['numpy', 'torch==1.7.1', 'scipy>=1.0.0', 'scikit-learn', 'matplotlib'],
      packages=find_packages())
