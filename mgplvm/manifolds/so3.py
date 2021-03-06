import numpy as np
from mgplvm import quaternion
from scipy import special
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from .base import Manifold
from typing import Tuple, Optional
from ..inducing_variables import InducingPoints


class So3(Manifold):

    # log of the uniform prior (negative log volume)
    log_uniform = (special.loggamma(2) - np.log(1) - 2 * np.log(np.pi))

    def __init__(self,
                 m: int,
                 d: Optional[int] = None,
                 mu: Optional[np.ndarray] = None,
                 Tinds: Optional[np.ndarray] = None):
        super().__init__(d=3)

        self.m = m
        self.d2 = 4  # dimensionality of the group parameterization

        # initialize at identity
        mudata = torch.tensor(np.array([[1, 0, 0, 0] for i in range(m)]),
                              dtype=torch.get_default_dtype())
        if mu is not None:
            mudata[Tinds, ...] = torch.tensor(mu,
                                              dtype=torch.get_default_dtype())

        self.mu = nn.Parameter(data=mudata, requires_grad=True)

        # per condition
        self.lprior_const = torch.tensor(
            special.loggamma(2) - np.log(1) - 2 * np.log(np.pi))

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
    def prms(self) -> Tensor:
        mu = self.mu
        norms = torch.norm(mu, dim=1, keepdim=True)
        return mu / norms

    @property
    def name(self):
        return 'So3(' + str(self.d) + ')'

    def lprior(self, g):
        return self.lprior_const * torch.ones(g.shape[:2])

    def transform(self, x: Tensor) -> Tensor:
        mu = self.prms
        return self.gmul(mu, x)  # group multiplication

    @staticmethod
    def expmap(x: Tensor, dim: int = -1) -> Tuple[Tensor, Tensor, Tensor]:
        theta = torch.norm(x, dim=dim, keepdim=True)
        v = x / theta
        y = torch.cat((torch.cos(theta), torch.sin(theta) * v), dim=dim)
        return y  # , theta, v

    @staticmethod
    def expmap_old(x: Tensor, dim: int = -1) -> Tuple[Tensor, Tensor, Tensor]:
        theta = torch.norm(x, dim=dim, keepdim=True)
        v = x / theta
        y = torch.cat((torch.cos(theta / 2), torch.sin(theta / 2) * v), dim=dim)
        return y  # , theta, v

    @staticmethod
    def expmap2(x: Tensor, dim: int = -1) -> Tensor:
        return F.normalize(x, dim=dim)

    @staticmethod
    def logmap(q: Tensor, dim: int = -1) -> Tensor:
        x = q[..., 0]
        y = torch.norm(q[..., 1:], dim=dim)
        theta = 2 * torch.atan2(y, x)
        return theta * (q[..., 1:] / y)

    @staticmethod
    def gmul(x: Tensor, y: Tensor) -> Tensor:
        return quaternion.product(x, y)

    @staticmethod
    def log_q(log_base_prob, x, d, kmax, dim=-1):
        '''
        theta = |x|/2
        '''

        theta = torch.norm(x, dim=dim, keepdim=True)
        v = x / theta
        ks = np.arange(-kmax, kmax + 1)
        zs = np.meshgrid(*(ks for _ in range(1)))
        zs = np.stack([z.flatten() for z in zs]).T * np.pi
        #zs = torch.from_numpy(zs).float().to(theta.device)
        zs = torch.tensor(zs, dtype=torch.get_default_dtype()).to(theta.device)
        theta = theta + zs[:, None, None, ...]  # (nk, n_b, m, 1)
        x = theta * v

        # |J|->1 as phi -> 0; cap at 1e-5 for numerical stability
        phi = 2 * theta + 1e-5
        l0 = torch.square(phi)
        l1 = 2 - 2 * torch.cos(phi)
        # |J^(-1)| = phi^2/(2 - 2*cos(phi))
        ljac = torch.log(l0) - torch.log(l1)

        lp = torch.logsumexp(log_base_prob(x) + ljac[..., 0], dim=0)
        return lp

    def log_q_old(log_base_prob, x, d, kmax, dim=-1):

        theta = torch.norm(x, dim=dim, keepdim=True)
        v = x / theta
        ks = np.arange(-kmax, kmax + 1)
        zs = np.meshgrid(*(ks for _ in range(1)))
        zs = np.stack([z.flatten() for z in zs]).T * 2 * np.pi
        #zs = torch.from_numpy(zs).float().to(theta.device)
        zs = torch.tensor(zs, dtype=torch.get_default_dtype()).to(theta.device)
        theta = theta + zs[:, None, None, ...]  # (nk, n_b, m, 1)
        x = theta * v

        # |J|->1 as theta -> 0; cap at 1e-5 for numerical stability
        theta = theta + 1e-5
        l0 = torch.square(theta)
        l1 = 2 - 2 * torch.cos(theta)
        # |J^(-1)| = theta^2/(2 - 2*cos(theta))
        ljac = torch.log(l0) - torch.log(l1)

        lp = torch.logsumexp(log_base_prob(x) + ljac[..., 0], dim=0)
        return lp

    @staticmethod
    def distance(x: Tensor, y: Tensor) -> Tensor:
        cosdist = (x[..., None] * y[..., None, :])
        cosdist = cosdist.sum(-3)
        return 4 * (1 - torch.square(cosdist))
