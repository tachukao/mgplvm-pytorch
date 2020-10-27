import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from .base import Manifold
from ..inducing_variables import InducingPoints
from typing import Optional


class Euclid(Manifold):

    def __init__(self,
                 m: int,
                 d: int,
                 mu: Optional[np.ndarray] = None,
                 Tinds: Optional[np.ndarray] = None):
        super().__init__(d)
        self.m = m
        self.d2 = d  # dimensionality of the group parameterization

        mudata = torch.randn(m, d) * 0.1
        if mu is not None:
            mudata[Tinds, ...] = torch.tensor(mu,
                                              dtype=torch.get_default_dtype())
        self.mu = nn.Parameter(data=mudata, requires_grad=True)

    def inducing_points(self, n, n_z, z=None):
        # distribute according to prior
        z = torch.randn(n, self.d, n_z) if z is None else z
        return InducingPoints(n, self.d, n_z, z=z)

    @property
    def prms(self) -> Tensor:
        return self.mu

    def transform(self, x: Tensor) -> Tensor:
        mu = self.prms
        return self.gmul(mu, x)

    def lprior(self, g):
        '''need empirical data here. g is (n_b x m x d)'''
        ps = -0.5 * torch.square(g) - 0.5 * np.log(2 * np.pi)
        return ps.sum(2)  # sum over d

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
    def distance_ard(x: Tensor, y: Tensor) -> Tensor:
        diff = x[..., None] - y[..., None, :]
        dist_sqr = torch.square(diff)
        return dist_sqr

    @property
    def name(self):
        return 'Euclid(' + str(self.d) + ')'
