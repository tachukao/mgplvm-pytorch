import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dists
from ..kernels import Kernel
from ..manifolds import Euclid
from ..manifolds.base import Manifold
from ..models import Svgp
from ..inducing_variables import InducingPoints
from ..likelihoods import Gaussian
from .common import Lprior
from typing import Optional


class LpriorEuclid(Lprior):
    def __init__(self, manif):
        if not isinstance(manif, Euclid):
            raise Exception("GP prior only works with Euclidean manifolds")

        super().__init__(manif)


class GP(LpriorEuclid):
    name = "GP"

    def __init__(self,
                 manif: Manifold,
                 kernel: Kernel,
                 n_z: Optional[int] = 20,
                 tmax: Optional[int] = 1):
        '''
        instantiate with a Kernel. This already specifies which parameters are learnable
        specifying tmax helps initialize the inducing points
        '''
        super().__init__(manif)
        d = manif.d
        assert (tmax is not None)
        assert (n_z is not None)
        #1d latent and n_z inducing points
        zinit = torch.linspace(0., tmax, n_z).reshape(1, 1, n_z)
        #separate inducing points for each latent dimension
        z = InducingPoints(d, 1, n_z, z=zinit.repeat(d, 1, 1))
        lik = Gaussian(
            d, variance=np.square(0.2), learn_sigma=False
        )  #.to(kernel.alpha.device) #consider fixing this to a small value as in GPFA
        self.svgp = Svgp(kernel, d, z, lik, whiten=True)  #construct svgp

    @property
    def prms(self):
        q_mu, q_sqrt, z = self.svgp.prms
        sigma_n = self.svgp.likelihood.prms
        return q_mu, q_sqrt, z, sigma_n

    def forward(self, x, ts, scale = 1.):
        '''
        x is a latent of shape (n_mc x n_samples x mx x d)
        ts is the corresponding timepoints of shape (n_samples x mx)
        '''
        n_mc, n_samples, T, d = x.shape
        # x now has shape (n_samples, n_mc*d, mx)
        x = x.permute(1, 0, 3, 2).reshape(n_samples, -1, T)

        # shape (d, n_mc)
        svgp_elbo = self.svgp.elbo(1, x, ts.reshape(1, n_samples, 1, -1), scale = scale)
        print(svgp_elbo.shape)
        return svgp_elbo.sum(-2)  #sum over dimensions

    @property
    def msg(self):
        ell = self.svgp.kernel.prms[1].mean()
        noise = self.svgp.likelihood.prms.sqrt()

        return (' prior ell {:.3f} | prior noise {:.3f} |').format(
            ell.item(), noise.item())
