# PyTorch implementation of mGPLVM
[Documentation](https://tachukao.github.io/mgplvm-pytorch) | [Examples](examples) | [Paper](https://papers.nips.cc/paper/2020/file/fedc604da8b0f9af74b6cfc0fab2163c-Paper.pdf)


## Setup

```sh
# inside virtual environment
pip install -e .
```

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

## Code used for NeurIPS 2020

The majority of the results in the NeurIPS 2020 paper "Manifold GPLVMs for discovering non-Euclidean latent structure in neural data" were generated using a Julia codebase which can be found [here](https://github.com/KrisJensen/mGPLVM).
This Julia codebase is somewhat slower, less flexible and less user friendly than the present PyTorch implementation but produces the results and plots from the paper.


## Running tests
```sh
py.test
```