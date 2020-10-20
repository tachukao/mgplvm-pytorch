# PyTorch implementation of mGPLVM

Examples can be found [here](examples).

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

The majority of the results in the NeurIPS 2020 paper "Manifold GPLVMs for discovering non-Euclidean latent structure in neural data" were generated using a Julia codebase which can be found [here](https://github.com/KrisJensen/mGPLVM_Neurips).
This Julia codebase is somewhat slower, less flexible and less user friendly than the present PyTorch implementation but produces the results and plots from the paper.
