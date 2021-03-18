import numpy as np
from . import quaternion
from scipy import special
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from .base import Manifold
from typing import Tuple, Optional, List
from ..inducing_variables import InducingPoints


class S3(Manifold):
    # log of the uniform prior (negative log volume)
    log_uniform = (special.loggamma(2) - np.log(2) - 2 * np.log(np.pi))

    def __init__(self, m: int, d: Optional[int] = None):
        """
        Parameters
        ----------
        m : int
            number of conditions/timepoints
        d : int
            latent dimensionality
        """
        super().__init__(d=3)

        self.m = m
        self.d2 = 4  # dimensionality of the group parameterization

        # per condition
        self.lprior_const = torch.tensor(
            special.loggamma(2) - np.log(2) - 2 * np.log(np.pi))

    @staticmethod
    def initialize(initialization, n_samples, m, d, Y):
        '''initializes latents - can add more exciting initializations as well'''
        # initialize at identity
        mudata = torch.zeros(n_samples, m, 4)
        mudata[:, :, 0] = 1
        return mudata

    def inducing_points(self, n, n_z, z=None):
        if z is None:
            z = torch.randn(n, self.d2, n_z)
            z = z / torch.norm(z, dim=1, keepdim=True)

        return InducingPoints(n,
                              self.d2,
                              n_z,
                              z=z,
                              parameterise=lambda x: self.expmap2(x, dim=-2))

    @property
    def name(self):
        return 'S(' + str(self.d) + ')'

    def lprior(self, g):
        return self.lprior_const * torch.ones(g.shape[:-1])

    @staticmethod
    def parameterise(x) -> Tensor:
        norms = torch.norm(x, dim=-1, keepdim=True)
        return x / norms

    @staticmethod
    def expmap(x: Tensor, dim: int = -1) -> Tensor:
        '''same as SO(3)'''
        theta = torch.norm(x, dim=dim, keepdim=True)
        v = x / theta
        y = torch.cat((torch.cos(theta), torch.sin(theta) * v), dim=dim)
        return y  # , theta, v

    @staticmethod
    def expmap2(x: Tensor, dim: int = -1) -> Tensor:
        return F.normalize(x, dim=dim)

    @staticmethod
    def logmap(q: Tensor, dim: int = -1) -> Tensor:
        '''same as SO(3)'''
        x = q[..., 0]
        y = torch.norm(q[..., 1:], dim=dim)
        theta = 2 * torch.atan2(y, x)
        return theta * (q[..., 1:] / y)

    @staticmethod
    def inverse(q: Tensor) -> Tensor:
        return quaternion.conj(q)

    @staticmethod
    def gmul(x: Tensor, y: Tensor) -> Tensor:
        '''same as SO(3)'''
        return quaternion.product(x, y)

    @staticmethod
    def log_q(log_base_prob, x, d, kmax, dim=-1):
        '''
        theta = |x|/2
        '''

        theta = torch.norm(x, dim=dim, keepdim=True)  #vector magintudes
        v = x / theta  #unit vectors
        ks = np.arange(-kmax, kmax + 1)
        zs = np.meshgrid(
            *(ks for _ in range(1)))  #construct equivalent elements
        zs = np.stack([z.flatten() for z in zs
                      ]).T * 2 * np.pi  #need to add multiples of 2pi
        zs = torch.tensor(zs, dtype=torch.get_default_dtype()).to(theta.device)
        theta = theta + zs[:, None, None, None,
                           ...]  # (nk, n_b, n_samples, m, 1)
        x = theta * v

        # |J|->1 as phi -> 0; cap at 1e-5 for numerical stability
        phi = 2 * theta + 1e-5  #magnitude of rotation
        l0 = torch.square(phi)
        l1 = 2 - 2 * torch.cos(phi)
        # |J^(-1)| = phi^2/(2 - 2*cos(phi)) = 2|x|^2/(1-cos(2|x|))
        ljac = torch.log(l0) - torch.log(l1)

        lp = torch.logsumexp(log_base_prob(x) + ljac[..., 0], dim=0)
        return lp

    @staticmethod
    def distance(x: Tensor, y: Tensor, ell: Optional[Tensor] = None) -> Tensor:
        # distance: 2 - 2 (x dot y)

        if ell is None:
            ell = torch.ones(1, 1, 1)

        z = x.transpose(-1, -2).matmul(y)
        res = 2 * (1 - z) / ell**2
        res.clamp_min_(0)
        return res
