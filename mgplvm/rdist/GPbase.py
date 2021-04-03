import torch
import numpy as np
from torch import nn, Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from ..utils import softplus, inv_softplus
from ..manifolds.base import Manifold
from .common import Rdist
from typing import Optional
from ..fast_utils.toeplitz import sym_toeplitz_matmul


class GPbase(Rdist):
    name = "GPbase"  # it is important that child classes have "GP" in their name, this is used in control flow

    def __init__(self,
                 manif: Manifold,
                 m: int,
                 n_samples: int,
                 ts: torch.Tensor,
                 _scale=0.9,
                 ell=None):
        """
        Parameters
        ----------
        manif: Manifold
            manifold of ReLie
        m : int
            number of conditions/timepoints
        n_samples: int
            number of samples
        ts: Tensor
            input timepoints for each sample (n_samples x 1 x m)
        mu : Optional[np.ndarray]
            initialization of the vartiational means (m x d2)
            
        Notes
        -----
        Our GP has prior N(0, K)
        We parameterize our posterior as N(K2 v, K2 I^2 K2)
        where K2 K2 = K and I(s) is some inner matrix which can take different forms.
        s is a vector of scale parameters for each time point.
        
        """

        super(GPbase, self).__init__(manif, 1)  #kmax = 1

        self.manif = manif
        self.d = manif.d
        self.m = m

        #initialize GP mean parameters
        nu = torch.randn((n_samples, self.d, m)) * 0.01
        self._nu = nn.Parameter(data=nu, requires_grad=True)  #m in the notes

        #initialize covariance parameters
        _scale = torch.ones(n_samples, self.d, m) * _scale  #n_diag x T
        self._scale = nn.Parameter(data=inv_softplus(_scale),
                                   requires_grad=True)

        #initialize length scale
        ell = (torch.max(ts) - torch.min(ts)) / 20 if ell is None else ell
        _ell = torch.ones(1, self.d, 1) * ell
        self._ell = nn.Parameter(data=inv_softplus(_ell), requires_grad=True)

        #pre-compute time differences (only need one row for the toeplitz stuff)
        self.ts = ts
        dts_sq = torch.square(ts - ts[..., :1])  #(n_samples x 1 x m)
        #sum over _input_ dimension, add an axis for _output_ dimension
        dts_sq = dts_sq.sum(-2)[:, None, ...]  #(n_samples x 1 x m)
        self.dts_sq = nn.Parameter(data=dts_sq, requires_grad=False)

        self.dt = (ts[0, 0, 1] - ts[0, 0, 0]).item()  #scale by dt

    @property
    def scale(self) -> torch.Tensor:
        return softplus(self._scale)

    @property
    def nu(self) -> torch.Tensor:
        return self._nu

    @property
    def ell(self) -> torch.Tensor:
        return softplus(self._ell)

    @property
    def prms(self):
        return self.nu, self.scale, self.ell

    @property
    def lat_mu(self):
        """return variational mean mu = K_half @ nu"""
        nu = self.nu
        K_half = self.K_half()  #(n_samples x d x m)
        mu = sym_toeplitz_matmul(K_half, nu[..., None])[..., 0]
        return mu.transpose(-1, -2)  #(n_samples x m x d)

    def K_half(self, sample_idxs=None):
        """compute one column of the square root of the prior matrix"""
        nu = self.nu  #mean parameters

        #K^(1/2) has length scale ell/sqrt(2) if K has ell
        ell_half = self.ell / np.sqrt(2)

        #K^(1/2) has sig var sig*2^1/4*pi^(-1/4)*ell^(-1/2) if K has sig^2 (1 x d x 1)
        sig_sqr_half = 1 * (2**(1 / 4)) * np.pi**(-1 / 4) * self.ell**(
            -1 / 2) * self.dt**(1 / 2)

        if sample_idxs is None:
            dts = self.dts_sq[:, ...]
        else:
            dts = self.dts_sq[sample_idxs, ...]

        # (n_samples x d x m)
        K_half = sig_sqr_half * torch.exp(-dts / (2 * torch.square(ell_half)))

        return K_half

    def I_v(self, v, sample_idxs=None):
        """
        Compute I @ v for some vector v.
        This should be implemented for each class separately.
        v is (n_samples x d x m x n_mc) where n_samples is the number of sample_idxs
        """
        pass

    def kl(self, batch_idxs=None, sample_idxs=None):
        """
        Compute KL divergence between prior and posterior.
        This should be implemented for each class separately
        """
        pass

    def full_cov(self):
        """Compute the full covariance Khalf @ I @ I @ Khalf"""
        v = torch.diag_embed(torch.ones(
            self._scale.shape))  #(n_samples x d x m x m)
        I = self.I_v(v)  #(n_samples x d x m x m)
        K_half = self.K_half()  #(n_samples x d x m)

        Khalf_I = sym_toeplitz_matmul(K_half, I)  #(n_samples x d x m x m)
        K_post = Khalf_I @ Khalf_I.transpose(-1, -2)  #Kpost = Khalf@I@I@Khalf

        return K_post.detach()

    def sample(self,
               size,
               Y=None,
               batch_idxs=None,
               sample_idxs=None,
               kmax=5,
               analytic_kl=False,
               prior=None):
        """
        generate samples and computes its log entropy
        """

        #compute KL analytically
        lq = self.kl(batch_idxs=batch_idxs,
                     sample_idxs=sample_idxs)  #(n_samples x d)

        K_half = self.K_half(sample_idxs=sample_idxs)  #(n_samples x d x m)
        n_samples, d, m = K_half.shape

        # sample a batch with dims: (n_samples x d x m x n_mc)
        v = torch.randn(n_samples, d, m, size[0])  # v ~ N(0, 1)
        #compute I @ v (n_samples x d x m x n_mc)
        I_v = self.I_v(v, sample_idxs=sample_idxs)

        nu = self.nu  #mean parameter (n_samples, d, m)
        if sample_idxs is not None:
            nu = nu[sample_idxs, ...]
        samp = nu[..., None] + I_v  #add mean parameter to each sample

        #compute K@(I@v+nu)
        x = sym_toeplitz_matmul(K_half, samp)  #(n_samples x d x m x n_mc)
        x = x.permute(-1, 0, 2, 1)  #(n_mc x n_samples x m x d)

        if batch_idxs is not None:  #only select some time points
            x = x[..., batch_idxs, :]

        #(n_mc x n_samples x m x d), (n_samples x d)
        return x, lq

    def gmu_parameters(self):
        return [self.nu]

    def concentration_parameters(self):
        return [self._scale, self._ell]

    def msg(self, Y=None, batch_idxs=None, sample_idxs=None):

        mu_mag = torch.sqrt(torch.mean(self.nu**2)).item()
        sig = torch.median(self.scale).item()
        ell = self.ell.mean().item()

        string = (' |mu| {:.3f} | sig {:.3f} | prior_ell {:.3f} |').format(
            mu_mag, sig, ell)

        return string
