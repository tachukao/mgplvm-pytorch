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
        if initialization in ['fa', 'FA']:
            #Y is N x m; reduce to d x m
            if Y is None:
                print('user must provide data for FA initialization')
            else:
                n = Y.shape[1]
                pca = decomposition.FactorAnalysis(n_components=d)
                Y = Y.transpose(0, 2, 1).reshape(n_samples * m, n)
                mudata = pca.fit_transform(Y)  #m*n_samples x d
                mudata *= 2 * np.pi / (np.amax(mudata) - np.amin(mudata))
                mudata = mudata.reshape(n_samples, m, d)
                return torch.tensor(mudata, dtype=torch.get_default_dtype())
        elif initialization in ['random', 'Random']:
            mudata = torch.randn(n_samples, m, d) * 0.1
            return mudata
        elif initialization in ['uniform_random']:
            mudata = torch.rand(n_samples, m, d) * 2 * np.pi
            return mudata
        else:
            print('initialization not recognized')
        return

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
    def distance(x: Tensor, y: Tensor, ell: Optional[Tensor] = None) -> Tensor:
        # distance = 2 - 2 cos(x-y)
        # here we use the identity: cox(x-y) = cos(x)cos(y) + sin(x)sin(y)
        d = x.shape[-2]

        if ell is None:
            ell = torch.ones(1, 1, 1)

        cx = torch.cos(x) / ell  #(... n x d x mx)
        cy = torch.cos(y) / ell
        sx = torch.sin(x) / ell
        sy = torch.sin(y) / ell
        z1_ = torch.cat([cx, sx], dim=-2)  #(... n x 2d x mx)
        z2_ = torch.cat([cy, sy], dim=-2)  #(... n x 2d x mx)

        const = d * ell.square().reciprocal().mean(
            -2)  # (1/n x 1/d x 1) -> (1/n x 1)

        res = 2 * (const[..., None] - z1_.transpose(-1, -2).matmul(z2_))
        res.clamp_min_(0)
        return res
