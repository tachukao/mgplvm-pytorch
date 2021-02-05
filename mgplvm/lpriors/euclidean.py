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
                 n,
                 m,
                 n_samples,
                 manif: Manifold,
                 kernel: Kernel,
                 ts: torch.Tensor,
                 n_z: int = 20,
                 d=1):
        """
        __init__ method for GP prior class (only works for Euclidean manif)
        Parameters
        ----------
        n : int
            number of output dimensions (i.e. dimensionality of the latent space)
        m : int
            number of time points
        n_samples : int 
            number of samples (each with a separate GP posterior)
        manif : mgplvm.manifolds.Manifold
            latent manifold
        kernel : mgplvm.kernels.kernel
            kernel used in the prior (does not haave to mtach the p(Y|G) kernel)
        ts: Tensor
            input timepoints for each sample (n_samples x d x m)
        n_z : Optional[int]
            number of inducing points used in the GP prior
        d : Optional[int]
            number of input dimensions -- defaults to 1 since the input is assumed to be time, but could also be other higher-dimensional observed variables.

        """
        super().__init__(manif)
        self.n = n
        self.m = m
        self.n_samples = n_samples
        self.d = d
        #1d latent and n_z inducing points
        zinit = torch.linspace(0., torch.max(ts).item(), n_z).reshape(1, 1, n_z)
        #separate inducing points for each latent dimension
        z = InducingPoints(n, d, n_z, z=zinit.repeat(n, d, 1))
        self.ts = ts
        #consider fixing this to a small value as in GPFA
        lik = Gaussian(n, sigma=0.2, learn_sigma=False)
        self.svgp = Svgp(kernel,
                         n,
                         m,
                         n_samples,
                         z,
                         lik,
                         whiten=True,
                         tied_samples=False)  #construct svgp

    @property
    def prms(self):
        q_mu, q_sqrt, z = self.svgp.prms
        sigma_n = self.svgp.likelihood.prms
        return q_mu, q_sqrt, z, sigma_n

    def forward(self, x, batch_idxs=None):
        '''
        x is a latent of shape (n_mc x n_samples x mx x d)
        ts is the corresponding timepoints of shape (n_samples x mx)
        '''
        n_mc, n_samples, m, n = x.shape
        assert (m == self.m)
        batch_size = m if batch_idxs is None else len(batch_idxs)
        ts = self.ts if batch_idxs is None else self.ts[..., batch_idxs]
        ts = ts.to(x.device)
        assert (n == self.n)
        # x now has shape (n_mc . n_samples , n, m)
        x = x.transpose(-1, -2)
        ts = ts.reshape(1, n_samples, self.d, -1).repeat(n_mc, 1, 1, 1)

        svgp_lik, svgp_kl = self.svgp.elbo(x, ts)

        # Here, we need to rescale the KL term so that it is over the batch not the full dataset, as that is what is expected in SVGPLVM
        elbo = (batch_size / m) * (svgp_lik - svgp_kl)

        # as the inducing points are shared across the full batch
        return elbo.sum(-1)  #sum over dimensions

    @property
    def msg(self):
        ell = self.svgp.kernel.prms[1].mean()
        noise = self.svgp.likelihood.prms.sqrt()

        return (' prior ell {:.3f} | prior noise {:.3f} |').format(
            ell.item(), noise.item())
