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
from sklearn import decomposition


class So3(Manifold):
    # log of the uniform prior (negative log volume)
    log_uniform = (special.loggamma(2) - np.log(1) - 2 * np.log(np.pi))

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
            special.loggamma(2) - np.log(1) - 2 * np.log(np.pi))

    def initialize(self, initialization, n_samples, m, d, Y):
        '''initializes latents - can add more exciting initializations as well'''
        if initialization in ['fa', 'FA']:
            #Y is N x m; reduce to d x m
            if Y is None:
                print('user must provide data for FA initialization')
            else:
                n = Y.shape[1]
                pca = decomposition.FactorAnalysis(n_components=d)
                Y = Y.transpose(0, 2, 1).reshape(n_samples * m, n)
                mudata = pca.fit_transform(Y)  #m*n_samples x d
                mudata *= 0.5 * np.pi / np.amax(
                    np.sqrt(np.sum(mudata**2, axis=-1)))
                mudata = torch.tensor(mudata, dtype=torch.get_default_dtype())
                mudata = self.expmap(mudata.reshape(n_samples, m, d))
                return mudata
        elif initialization in ['random', 'Random']:
            # initialize at identity
            mudata = self.expmap(torch.randn(n_samples, m, 3) * 0.1)
            #mudata = torch.tensor(np.array([[1, 0, 0, 0] for i in range(m)]), dtype=torch.get_default_dtype())
            return mudata
        else:
            print('initialization not recognized')
        return

    def parameterise_inducing(self, x):
        return self.expmap2(x, dim=-2)

    def inducing_points(self, n, n_z, z=None):
        if z is None:
            z = torch.randn(n, self.d2, n_z)
            z = z / torch.norm(z, dim=1, keepdim=True)

        return InducingPoints(n,
                              self.d2,
                              n_z,
                              z=z,
                              parameterise=self.parameterise_inducing)
        #parameterise=lambda x: self.expmap2(x, dim=-2))

    @property
    def name(self):
        return 'So3(' + str(self.d) + ')'

    def lprior(self, g):
        return self.lprior_const * torch.ones(g.shape[:-1])

    @staticmethod
    def parameterise(x) -> Tensor:
        norms = torch.norm(x, dim=-1, keepdim=True)
        return x / norms

    @staticmethod
    def expmap(x: Tensor, dim: int = -1, jitter=1e-8) -> Tensor:
        '''
        x \\in R^3 -> q \\in R^4 s.t. ||q|| = 1
        '''
        x = x + (jitter * torch.randn(x.shape)).to(x.device)  #avoid nans
        theta = torch.norm(x, dim=dim, keepdim=True)
        v = x / theta
        y = torch.cat((torch.cos(theta), torch.sin(theta) * v), dim=dim)
        return y  # , theta, v

    @staticmethod
    def expmap2(x: Tensor, dim: int = -1) -> Tensor:
        return F.normalize(x, dim=dim)

    @staticmethod
    def logmap(q: Tensor, dim: int = -1) -> Tensor:
        '''
        q \\in R^4 s.t. ||q|| = 1 -> x \\in R^3
        '''
        #make first index positive as convention -- this gives theta \in [0, pi] and u on the hemisphere
        q = torch.sign(q[..., :1]) * q
        a = q[..., :1]
        theta = 2 * torch.acos(a)  #magnitude of rotation; ||x|| = theta/2
        u = q[..., 1:] / torch.norm(q[..., 1:], dim=dim, keepdim=True)
        return 0.5 * theta * u

    @staticmethod
    def inverse(q: Tensor) -> Tensor:
        return quaternion.conj(q)

    @staticmethod
    def gmul(x: Tensor, y: Tensor) -> Tensor:
        return quaternion.product(x, y)

    @staticmethod
    def log_q(log_base_prob, x, d, kmax, dim=-1):
        '''
        phi = |x|/2
        '''

        theta = torch.norm(x, dim=dim, keepdim=True)
        v = x / theta
        ks = np.arange(-kmax, kmax + 1)
        zs = np.meshgrid(*(ks for _ in range(1)))
        zs = np.stack([z.flatten() for z in zs]).T * np.pi
        zs = torch.tensor(zs, dtype=torch.get_default_dtype()).to(theta.device)
        theta = theta + zs[:, None, None, None,
                           ...]  # (nk, n_b, n_samples, m, 1)
        x = theta * v

        # |J|->1 as phi -> 0; cap at 1e-5 for numerical stability
        phi = 2 * theta + 1e-5
        l0 = torch.square(phi)
        l1 = 2 - 2 * torch.cos(phi)
        # |J^(-1)| = phi^2/(2 - 2*cos(phi))
        ljac = torch.log(l0) - torch.log(l1)

        lp = torch.logsumexp(log_base_prob(x) + ljac[..., 0], dim=0)
        return lp

    @staticmethod
    def distance(x: Tensor, y: Tensor, ell: Optional[Tensor] = None) -> Tensor:
        """x, y: (..., n x d x m)"""
        # distance: 4 - 4 (x dot y)^2

        if ell is None:
            ell = torch.ones(1, 1, 1)

        z = x.transpose(-1, -2).matmul(y)  # (..., n, m, m)
        res = 4 * (1 - z.square()) / ell**2
        res.clamp_min_(0)
        return res
