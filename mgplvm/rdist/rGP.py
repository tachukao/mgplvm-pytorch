import torch
import numpy as np
from torch import nn, Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from ..utils import softplus, inv_softplus
from ..manifolds.base import Manifold
from .common import Rdist
from typing import Optional
from ..fast_utils.toeplitz import sym_toeplitz_matmul, sym_toeplitz


class EP_GP(Rdist):
    name = "EP_GP"

    def __init__(self,
                 manif: Manifold,
                 m: int,
                 n_samples: int,
                 ts: torch.Tensor,
                 mu=None,
                 initialization: Optional[str] = 'random',
                 Y=None,
                 _scale=0.2,
                 ell=None,
                 use_fast_toeplitz=True):
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
        intialization : Optional[str]
            string to specify type of initialization
            ('random'/'PCA'/'identity' depending on manifold)
        mu : Optional[np.ndarray]
            initialization of the vartiational means (m x d2)
        Y : Optional[np.ndarray]
            data used to initialize latents (n x m)
            
        Notes
        -----
        """

        super(EP_GP, self).__init__(manif, 1)

        self.use_fast_toeplitz = use_fast_toeplitz
        self.manif = manif
        self.d = manif.d

        #initialize GP mean
        nu = torch.randn((n_samples, self.d, m)) * 1
        self._nu = nn.Parameter(data=nu, requires_grad=True)  #m in the notes

        #self._nu_s = nn.Parameter(data=torch.ones(1), requires_grad=True)
        #self._nu_i = nn.Parameter(data=torch.zeros(1), requires_grad=True)

        #initialize covariance parameters
        _scale = torch.ones(n_samples, self.d, m) * _scale * (
            1 + torch.randn(m) / 100)  #n_diag x T
        self._scale = nn.Parameter(data=inv_softplus(_scale),
                                   requires_grad=True)

        ell = (torch.max(ts) - torch.min(ts)) / 20 if ell is None else ell
        _ell = torch.ones(1, self.d, 1) * ell
        self._ell = nn.Parameter(data=inv_softplus(_ell), requires_grad=True)

        #pre-compute time differences (only need one row for the toeplitz stuff)
        self.ts = ts
        self.dts_sq = torch.square(ts - ts[..., :1])  #(n_samples x 1 x m)
        #sum over _input_ dimension, add an axis for _output_ dimension
        self.dts_sq = self.dts_sq.sum(-2)[:, None, ...]  #(n_samples x 1 x m)

    @property
    def scale(self) -> torch.Tensor:
        return softplus(self._scale)

    @property
    def nu(self) -> torch.Tensor:
        return self._nu
        #return self._nu_s * ( (self._nu - self._nu.mean()) / self._nu.std()) + self._nu_i

    @property
    def ell(self) -> torch.Tensor:
        return softplus(self._ell)

    @property
    def prms(self):
        nu = self.nu  #mean parameters

        ell_half = self.ell / np.sqrt(
            2)  #K^(1/2) has length scale ell/sqrt(2) if K has ell

        #K^(1/2) has sig variance sigma*2^1/4*pi^(-1/4)*ell^(-1/2) if K has sigma^2
        sig_sqr_half = 1 * (2**(1 / 4)) * np.pi**(-1 / 4) * (self.ell**(-1 / 2)
                                                            )  #(1 x d x 1)
        
        # (n_samples x d x m)
        K_half = sig_sqr_half * torch.exp(-self.dts_sq.to(ell_half.device) / (2 * torch.square(ell_half)))

        # the if and else do the same matmul, but sym_toeplitz takes advantage of structure
        if self.use_fast_toeplitz:
            #(n_samples x d x m x 1)
            mu = sym_toeplitz_matmul(K_half, nu[...,None])

        else:
            #compute full TxT covariance matrix
            #(n_samples x d x m x m)
            K_half = sym_toeplitz(K_half)
            mu = K_half @ nu[..., None]  #(n_samples x d x m x 1)

        return mu[..., 0].transpose(-1, -2), K_half

    def mvn(self, L, mu=None):
        """
        L is loower cholesky factor
        """
        n_samples, _, _, m = L.shape
        mu = torch.zeros(n_samples, self.d, m).to(
            L.device) if mu is None else mu
        return MultivariateNormal(mu, scale_tril=L)

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

        #(n_samples x m x d), (n_samples x d x m x m)/(n_samples x d x m)
        mu, K_half = self.lat_prms(batch_idxs=batch_idxs,
                                   sample_idxs=sample_idxs)

        # sample a batch with dims: (n_samples x d x m x n_mc)
        rand = torch.randn((mu.shape[0], mu.shape[2], mu.shape[1], size[0]))

        #multiply by diagonal scale
        scale = self.scale  # (n_samples, d, m)
        Sv = scale[..., None] * rand.to(
            scale.device)  #(n_samples x d x m x n_mc) S*v

        # the if and else do the same matmul, but sym_toeplitz takes advantage of structure
        if self.use_fast_toeplitz:
            x = sym_toeplitz_matmul(K_half, Sv)
        else:
            x = K_half @ Sv

        x = x.permute(-1, 0, 2, 1)  #(n_mc x n_samples x m x d)
        x = x + mu[None, ...]  #add mean

        #(n_mc x n_samples x m x d), (n_samples x d)
        return x, lq

    def kl(self, batch_idxs=None, sample_idxs=None):
        """
        compute KL divergence analytically
        """
        #(n_samples x d x m), (n_samples x d x m)
        nu, S = self.nu, self.scale
        if batch_idxs is not None:
            nu = nu[..., batch_idxs]
            S = S[..., batch_idxs]
        if sample_idxs is not None:
            nu = nu[sample_idxs, ...]
            S = S[sample_idxs, ...]

        TrTerm = torch.square(S).sum(-1)  #(n_samples x d)
        MeanTerm = torch.square(nu).sum(-1)  #(n_samples x d)
        DimTerm = S.shape[-1]
        LogTerm = 2 * (torch.log(S)).sum(-1)  #(n_samples x d)

        kl = 0.5 * (TrTerm + MeanTerm - DimTerm - LogTerm)
        return kl

    def gmu_parameters(self):
        return [self.nu]

    def concentration_parameters(self):
        return [self._scale, self._ell]

    def lat_prms(self, Y=None, batch_idxs=None, sample_idxs=None):
        mu, K_half = self.prms
        if batch_idxs is not None:
            mu = mu[:, batch_idxs, :]
            if self.use_fast_toeplitz:
                K_half = K_half[..., batch_idxs]
            else:
                K_half = K_half[..., :, batch_idxs][..., batch_idxs, :]
        if sample_idxs is not None:
            mu = mu[sample_idxs, ...]
            K_half = K_half[sample_idxs, ...]

        return mu, K_half

    def lat_gmu(self, Y=None, batch_idxs=None, sample_idxs=None):
        return self.lat_prms(Y=Y,
                             batch_idxs=batch_idxs,
                             sample_idxs=sample_idxs)[0]

    def lat_gamma(self, Y=None, batch_idxs=None, sample_idxs=None):
        return self.lat_prms(Y=Y,
                             batch_idxs=batch_idxs,
                             sample_idxs=sample_idxs)[1]

    def msg(self, Y=None, batch_idxs=None, sample_idxs=None):
        mu, _ = self.lat_prms(Y=Y,
                              batch_idxs=batch_idxs,
                              sample_idxs=sample_idxs)

        mu_mag = torch.sqrt(torch.mean(mu**2)).item()
        sig = torch.median(self.scale).item()

        ell = self.ell.mean().item()
        string = (' |mu| {:.3f} | sig {:.3f} | prior_ell {:.3f} |').format(
            mu_mag, sig, ell)
        return string
