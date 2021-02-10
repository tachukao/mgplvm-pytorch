import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from .base import Manifold
from ..inducing_variables import InducingPoints
from typing import Optional, List
from sklearn import decomposition


class Euclid(Manifold):

    def __init__(self, m: int, d: int):
        """
        Parameters
        ----------
        m : int
            number of conditions/timepoints
        d : int
            latent dimensionality
        """
        super().__init__(d)
        self.m = m
        self.d2 = d  # dimensionality of the group parameterization

    @staticmethod
    def initialize(initialization, n_samples, m, d, Y):
        '''initializes latents - can add more exciting initializations as well
        Y is (n_samples x n x m)'''
        if initialization in ['fa', 'FA']:
            #Y is n_samples x n x m; reduce to n_samples x m x d
            if Y is None:
                print('user must provide data for FA initialization')
            else:
                n = Y.shape[1]
                pca = decomposition.FactorAnalysis(n_components=d)
                #pca = decomposition.PCA(n_components=d)
                Y = Y.transpose(0, 2, 1).reshape(n_samples * m, n)
                mudata = pca.fit_transform(Y)  #m*n_samples x d
                mudata = 0.5 * mudata / np.std(mudata, axis=0,
                                               keepdims=True)  #normalize
                mudata = mudata.reshape(n_samples, m, d)
                return torch.tensor(mudata, dtype=torch.get_default_dtype())
        elif initialization in ['random', 'Random']:
            mudata = torch.randn(n_samples, m, d) * 0.1
            return mudata
        else:
            print('initialization not recognized')
        return

    def inducing_points(self, n, n_z, z=None):
        # distribute according to prior
        z = torch.randn(n, self.d, n_z) if z is None else z
        return InducingPoints(n, self.d, n_z, z=z)

    def lprior(self, g):
        '''need empirical data here. g is (n_b x n_samples x m x d)'''
        ps = -0.5 * torch.square(g) - 0.5 * np.log(2 * np.pi)
        return ps.sum(2)  # sum over d

    @staticmethod
    def parameterise(x) -> Tensor:
        return x

    @staticmethod
    def log_q(log_base_prob, x, d=None, kmax=None):
        lp = log_base_prob(x)
        return lp

    @staticmethod
    def expmap(x: Tensor) -> Tensor:
        return x

    @staticmethod
    def logmap(x: Tensor) -> Tensor:
        return x

    @staticmethod
    def inverse(x: Tensor) -> Tensor:
        return -x

    @staticmethod
    def gmul(x: Tensor, y: Tensor) -> Tensor:
        return x + y

    @staticmethod
    def distance(x: Tensor, y: Tensor, ell: Optional[Tensor] = None) -> Tensor:
        # Based on implementation here: https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/kernels/kernel.py

        #scale lengths by ell
        if ell is not None:
            x = x / ell
            y = y / ell

        # Compute squared distance matrix using quadratic expansion
        x_norm = x.pow(2).sum(dim=-2, keepdim=True)
        x_pad = torch.ones_like(x_norm)
        y_norm = y.pow(2).sum(dim=-2, keepdim=True)
        y_pad = torch.ones_like(y_norm)
        x_ = torch.cat([-2.0 * x, x_norm, x_pad], dim=-2)
        y_ = torch.cat([y, y_pad, y_norm], dim=-2)
        res = x_.transpose(-1, -2).matmul(y_)

        # Zero out negative values
        res.clamp_min_(0)
        return res

    @property
    def name(self):
        return 'Euclid(' + str(self.d) + ')'
