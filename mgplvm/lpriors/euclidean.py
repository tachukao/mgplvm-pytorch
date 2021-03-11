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
        batch_size = m if batch_idxs is None else len(batch_idxs)
        ts = self.ts if batch_idxs is None else self.ts[..., batch_idxs]
        ts = ts.to(x.device)
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


class GP_full(LpriorEuclid):
    name = "GP"

    def __init__(self, n, m, n_samples, manif: Manifold, ts: torch.Tensor, d=1):
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
        ts: Tensor
            input timepoints for each sample (n_samples x d x m)
        d : Optional[int]
            number of input dimensions -- defaults to 1 since the input is assumed to be time, but could also be other higher-dimensional observed variables.

        """
        super().__init__(manif)
        self.n = n
        self.m = m
        self.n_samples = n_samples
        self.d = d
        self.ts = ts

        jitter = 1e-3  #noise std
        self.scale = np.sqrt(
            1 - jitter**2
        )  #nn.Parameter(torch.ones(1, n, 1)*np.sqrt(1-jitter**2), requires_grad=False)
        #self.noise = nn.Parameter(torch.ones(n_samples, n, m)*jitter, requires_grad=False)
        self.noise = jitter

        ell = (torch.max(ts) - torch.min(ts)) / 10
        _ell = torch.ones(1, n, 1, 1) * ell
        self._ell = nn.Parameter(data=inv_softplus(_ell),
                                 requires_grad=True)  #True)

        self.dts_sq = torch.square(ts[..., None] -
                                   ts[..., None, :])  #(n_samples x 1 x m x m)
        self.dts_sq = self.dts_sq.sum(
            -3
        )[:, None,
          ...]  #sum over _input_ dimension, add an axis for _output_ dimension
        #print('prior dts:', self.dts_sq.shape)

    def mvn(self, L, batch_idxs=None):
        """
        L is lower cholesky factor
        """
        L = L if batch_idxs is None else L[..., batch_idxs][..., batch_idxs, :]
        n_samples, d, _, m = L.shape  #n_samples x d x m x m
        mu = torch.zeros(n_samples, d, m).to(L.device)
        return dists.MultivariateNormal(mu, scale_tril=L)

    @property
    def ell(self) -> torch.Tensor:
        return softplus(self._ell)

    @property
    def prms(self):
        """
        return covariance matrix
        """
        gamma = torch.exp(
            -self.dts_sq /
            (2 * torch.square(self.ell)))  #(n_samples x d x m x m)
        #gamma = self.scale[..., None, :] * gamma * self.scale[..., None]
        gamma = (self.scale**2) * gamma
        #gamma = gamma + torch.diag_embed(torch.square(self.noise)) #covariance
        gamma = gamma + torch.eye(gamma.shape[-1]).to(
            gamma.device) * (self.noise**2)
        #print('prior gamma:', gamma.shape)
        return gamma

    def forward(self, x, batch_idxs=None):
        '''
        x is a latent of shape (n_mc x n_samples x mx x d)
        '''

        n_mc, n_samples, m, n = x.shape  #n is the output dimensionality
        assert (m == self.m)
        assert (n == self.n)
        #batch_size = m if batch_idxs is None else len(batch_idxs)

        gamma = self.prms
        Lp = torch.cholesky(gamma)
        p = self.mvn(Lp)
        lp = p.log_prob(x.transpose(-1, -2))
        #print('prior logprob:', lp.shape) #(n_mc x n_samples x n)

        return lp.sum(-1).sum(-1)  #sum over dimensions and samples (n_mc)

    @property
    def msg(self):
        ell = self.ell
        return (' prior ell {:.3f} | ').format(ell.mean().item())


def fio_id(x):
    return x


def fio_ReLU(x):
    return torch.max(0, x)


def fio_tanh(x):
    return torch.tanh(x)


class DS(LpriorEuclid):

    def __init__(
        self,
        manif: Manifold,
        fio=fio_id,
        fix_noise_scale=True,
    ):
        """
        x_t = f(A*x_(t-1)) + N(0, Q)
        both A and B are full matrices
        f can be the identity (default; LDS prior) or some non-linear function.
        """
        super().__init__(manif)
        d = self.d
        A = torch.diag_embed(torch.ones(d))
        Q = torch.diag_embed(torch.ones(d))

        self.A = nn.Parameter(data=A, requires_grad=True)
        self.Q = nn.Parameter(data=Q, requires_grad=True)

        self.fix_noise_scale = fix_noise_scale

        return

    @property
    def prms(self):
        A = self.A
        Q = self.Q

        #fix the scale of Q to avoid degeneracies?
        #here we do this using the trace but there are probably better ways
        if self.fix_noise_scale:
            Q = Q / torch.diag(Q).mean() * 0.5
        return self.A, self.Q

    def forward(self, x, batch_idxs=None):
        """
        x: (n_mc, n_samples, m, d)
        """

        A, Q = self.prms
        #print(A.shape, Q.shape, x.shape)
        xA = torch.matmul(x, A)  #(n_mc, n_samples, m, d)
        dx = x[..., 1:, :] - xA[..., :-1, :]
        #print(dx.shape)

        mu = torch.zeros(self.d).to(x.device)
        normal = dists.MultivariateNormal(mu, covariance_matrix=Q)
        lq = normal.log_prob(dx)  #(n_mc x n_samplesx m-1)
        #print('LDS logprob shape:', lq.shape)
        lq = lq.sum(-1).sum(-1)  #(n_mc)
        #print('LDS prior shape:', lq.shape)

        #in the future, we may want an explicit prior over the initial point
        return lq

    @property
    def msg(self):
        return ''
        ar_c, ar_phi, ar_eta = self.prms
        lp_msg = (' ar_c {:.3f} | ar_phi_avg {:.3f} | ar_eta {:.3f} |').format(
            ar_c.detach().cpu().mean(),
            ar_phi.detach().cpu().mean(),
            ar_eta.detach().cpu().sqrt().mean())
        return lp_msg
