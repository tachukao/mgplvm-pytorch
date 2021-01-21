import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from .base import Manifold
from ..inducing_variables import InducingPoints
from typing import Optional
from sklearn import decomposition


class Torus(Manifold):
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

        # per condition
        self.lprior_const = torch.tensor(-self.d * np.log(2 * np.pi))

    @staticmethod
    def initialize(initialization, n_samples, m, d, Y):
        '''initializes latents - can add more exciting initializations as well'''
        if initialization == 'pca':
            #Y is N x m; reduce to d x m
            if Y is None:
                print('user must provide data for PCA initialization')
            else:
                Y = Y.reshape(-1, m)
                pca = decomposition.PCA(n_components=d)
                mudata = pca.fit_transform(Y.T)  #m x d
                #constrain to injectivity radius
                mudata = mudata * 2 * np.pi / (np.amax(mudata) -
                                               np.amin(mudata))
                mudata = mudata.reshape(Y.shape[0], -1, d)
                return torch.tensor(mudata, dtype=torch.get_default_dtype())
        mudata = torch.randn(n_samples, m, d) * 0.1
        return mudata

    def inducing_points(self, n, n_z, z=None):
        z = torch.rand(n, self.d, n_z) * 2 * np.pi if z is None else z
        return InducingPoints(n, self.d, n_z, z=z)

    def lprior(self, g: Tensor) -> Tensor:
        return self.lprior_const * torch.ones(g.shape[:-1])

    # log of the uniform prior (negative log volume) for T^d
    @property
    def log_uniform(self) -> Tensor:
        return -self.d * np.log(2 * np.pi)

    @property
    def name(self):
        return 'Torus(' + str(self.d) + ')'

    @staticmethod
    def parameterise(x) -> Tensor:
        return x

    @staticmethod
    def expmap(x: Tensor) -> Tensor:
        '''move to [-pi, pi]'''
        return (x + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def logmap(x: Tensor) -> Tensor:
        '''move to [-pi, pi]'''
        return (x + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def log_q(log_base_prob, x, d, kmax):
        """ log of the variational distribution (~-H(Q))"""
        ks = np.arange(-kmax, kmax + 1)
        zs = np.meshgrid(*(ks for _ in range(d)))
        zs = np.stack([z.flatten() for z in zs]).T * 2. * np.pi
        zs = torch.tensor(zs, dtype=torch.get_default_dtype())
        zs = zs.to(x.device)  # meshgrid shape (2kmax+1)^n
        y = x + zs[:, None, None, None,
                   ...]  # meshgrid x n_b x n_samples, m x d
        lp = torch.logsumexp(log_base_prob(y), dim=0)  # n_b x n_samples, m
        return lp

    @staticmethod
    def inverse(x: Tensor) -> Tensor:
        return -x

    @staticmethod
    def gmul(x: Tensor, y: Tensor) -> Tensor:
        return x + y

    @staticmethod
    def distance(x: Tensor, y: Tensor) -> Tensor:
        diff = 2 - (2 * torch.cos(x[..., None] - y[..., None, :]))
        dist_sqr = torch.sum(diff, dim=-3)
        return dist_sqr

    @staticmethod
    def linear_distance(x: Tensor, y: Tensor) -> Tensor:
        dist = torch.cos(x[..., None] - y[..., None, :]).sum(dim=-3)
        return dist

    @staticmethod
    def distance_ard(x: Tensor, y: Tensor) -> Tensor:
        return 2 - (2 * torch.cos(x[..., None] - y[..., None, :]))
