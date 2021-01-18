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
    def initialize(initialization, m, d, Y):
        '''initializes latents - can add more exciting initializations as well'''
        if initialization == 'pca':
            #Y is N x m; reduce to m x d
            if Y is None:
                print('user must provide data for PCA initialization')
            else:
                pca = decomposition.PCA(n_components=d)
                mudata = pca.fit_transform(Y[:, :, 0].T)  #m x d
                mudata = mudata / np.std(mudata, axis=0, keepdims=True)
                return torch.tensor(mudata, dtype=torch.get_default_dtype())
        # default initialization
        mudata = torch.randn(m, d) * 0.1
        return mudata

    def inducing_points(self, n, n_z, z=None):
        # distribute according to prior
        z = torch.randn(n, self.d, n_z) if z is None else z
        return InducingPoints(n, self.d, n_z, z=z)

    def lprior(self, g):
        '''need empirical data here. g is (n_b x m x d)'''
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
    def distance(x: Tensor, y: Tensor) -> Tensor:
        diff = x[..., None] - y[..., None, :]
        dist_sqr = torch.sum(torch.square(diff), dim=-3)
        return dist_sqr

    @staticmethod
    def linear_distance(x: Tensor, y: Tensor) -> Tensor:
        dist = (x[..., None] * y[..., None, :]).sum(dim=-3)
        return dist

    @staticmethod
    def distance_ard(x: Tensor, y: Tensor) -> Tensor:
        diff = x[..., None] - y[..., None, :]
        dist_sqr = torch.square(diff)
        return dist_sqr

    @property
    def name(self):
        return 'Euclid(' + str(self.d) + ')'
