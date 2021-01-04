import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from .base import Manifold
from ..inducing_variables import InducingPoints
from typing import Optional
from sklearn import decomposition


class Torus(Manifold):
    def __init__(self,
                 m: int,
                 d: int,
                 mu: Optional[np.ndarray] = None,
                 Tinds: Optional[np.ndarray] = None,
                 initialization: Optional[str] = 'random',
                 Y: Optional[np.ndarray] = None):
        super().__init__(d)
        self.m = m
        self.d2 = d  # dimensionality of the group parameterization

        mudata = self.initialize(initialization, m, d, Y)
        if mu is not None:
            mudata[Tinds, ...] = torch.tensor(mu,
                                              dtype=torch.get_default_dtype())

        self.mu = nn.Parameter(data=mudata, requires_grad=True)

        # per condition
        self.lprior_const = torch.tensor(-self.d * np.log(2 * np.pi))

    @staticmethod
    def initialize(initialization, m, d, Y):
        '''initializes latents - can add more exciting initializations as well'''
        if initialization == 'pca':
            #Y is N x m; reduce to d x m
            if Y is None:
                print('user must provide data for PCA initialization')
            else:
                pca = decomposition.PCA(n_components=d)
                mudata = pca.fit_transform(Y.T)  #m x d
                mudata *= 2 * np.pi / (np.amax(mudata) - np.amin(mudata))
                return torch.tensor(mudata, dtype=torch.get_default_dtype())
        mudata = torch.randn(m, d) * 0.1
        return mudata

    def inducing_points(self, n, n_z, z=None):
        z = torch.rand(n, self.d, n_z) * 2 * np.pi if z is None else z
        return InducingPoints(n, self.d, n_z, z=z)

    @property
    def prms(self) -> Tensor:
        return self.mu

    def lprior(self, g: Tensor) -> Tensor:
        return self.lprior_const * torch.ones(g.shape[:2])

    def transform(self, x, mu=None, batch_idxs=None):
        mu = self.prms
        if batch_idxs is not None:
            mu = mu[batch_idxs]
        return self.gmul(mu, x)

    # log of the uniform prior (negative log volume) for T^d
    @property
    def log_uniform(self) -> Tensor:
        return -self.d * np.log(2 * np.pi)

    @property
    def name(self):
        return 'Torus(' + str(self.d) + ')'

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
        zs = torch.from_numpy(zs).float()
        zs = zs.to(x.device)  # meshgrid shape (2kmax+1)^n
        y = x + zs[:, None, None, ...]  # meshgrid x n_b x m x n_samples
        lp = torch.logsumexp(log_base_prob(y), dim=0)  # n_b x m
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
