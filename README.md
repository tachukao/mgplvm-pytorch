# PyTorch implementation of mGPLVM and bGPFA

[![CI](https://github.com/tachukao/mgplvm-pytorch/workflows/CI/badge.svg?branch=develop)](https://github.com/tachukao/mgplvm-pytorch/actions?query=workflow%3ACI) | 
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://tachukao.github.io/mgplvm-pytorch) | 
[Examples](examples) | 
[mGPLVM paper](https://papers.nips.cc/paper/2020/file/fedc604da8b0f9af74b6cfc0fab2163c-Paper.pdf)
[bGPFA paper](https://www.biorxiv.org/content/10.1101/2021.06.03.446788v1)

This repository contains code for running both the manifold GPLVM (Jensen et al. 2020) and Bayesian GPFA (Jensen and Kao et al. 2021) and is currently still in active development.
Currently, the master branch can be used for mGPLVM and the bGPFA branch for bGPFA.

## Setup

```sh
# inside virtual environment
pip install -e .
```

To run on GPU, it may be necessary to first install pytorch with GPU support.

## Dependencies

- pytorch
- numpy
- scipy
- matplotlib

## Autoformat

This library uses [yapf](https://github.com/google/yapf) for autoformatting.
To autoformat all files in this directory:

```sh
yapf -ir .
```

## Running tests
```sh
py.test
```
