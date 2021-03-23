import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dists
from torch.distributions import transform_to, constraints
from ..kernels import Kernel
from ..manifolds import Euclid
from ..manifolds.base import Manifold
from ..models import Svgp
from ..inducing_variables import InducingPoints
from ..likelihoods import Gaussian
from .common import Lprior
from ..utils import softplus, inv_softplus
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
                 d=1,
                 learn_sigma=False):
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
        self.lik = Gaussian(n,
                            sigma=torch.ones(n) * 0.2,
                            learn_sigma=learn_sigma)
        self.svgp = Svgp(kernel,
                         n,
                         m,
                         n_samples,
                         z,
                         self.lik,
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
        batch_size = m
        ts = self.ts.to(x.device)
        assert (n == self.n)
        # x now has shape (n_mc, n_samples , n, m)
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
        noise = self.lik.sigma.mean()

        return (' prior ell {:.3f} | prior noise {:.3f} |').format(
            ell.item(), noise.item())


def fio_id(x):
    return x


def fio_ReLU(x):
    return torch.nn.functional.relu(x)


def fio_tanh(x):
    return torch.tanh(x)


class DS(LpriorEuclid):
    name = "DS"

    def __init__(
        self,
        manif: Manifold,
        fio=fio_id,
    ):
        """
        x_t = f(A*x_(t-1)) + N(0, Q)
        where A is Hurwitz and Q is diagonal
        f can be the identity (default; LDS prior) or some non-linear function.
        """
        super().__init__(manif)
        d = self.d

        Q = torch.diag_embed(torch.ones(d) * 0.5)
        self.Q = nn.Parameter(data=Q, requires_grad=False
                             )  # fixes the scale and orientation of the latents

        A = torch.diag_embed(torch.ones(d))
        self.A = nn.Parameter(data=A, requires_grad=True)
        print('initialized DS')

    @property
    def prms(self):
        O, R = torch.qr(self.A)
        signs = torch.diag_embed(torch.sign(torch.diag(R)))
        O = O @ signs
        R = signs @ R

        Lsqrt = torch.diag(R)
        L_I = torch.sqrt(torch.square(Lsqrt) + 1)**(-1)

        A = torch.diag_embed(Lsqrt) @ O @ torch.diag_embed(L_I)
        Q = self.Q
        return A, Q

    def forward(self, x, batch_idxs=None):
        """
        x: (n_mc, n_samples, m, d)
        """

        A, Q = self.prms
        xA = torch.matmul(x, A)  #(n_mc, n_samples, m, d)
        dx = x[..., 1:, :] - xA[..., :-1, :]

        mu = torch.zeros(self.d).to(x.device)
        normal = dists.MultivariateNormal(mu, scale_tril=Q)
        lq = normal.log_prob(dx)  #(n_mc x n_samplesx m-1)
        lq = lq.sum(-1).sum(-1)  #(n_mc)

        #in the future, we may want an explicit prior over the initial point
        return lq

    @property
    def msg(self):
        A, Q = self.prms
        lp_msg = (' A {:.3f} |').format(torch.diag(A).mean().item())
        return lp_msg
