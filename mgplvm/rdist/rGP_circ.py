import torch
import numpy as np
from torch import nn, Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from ..utils import softplus, inv_softplus
from ..manifolds.base import Manifold
from .common import Rdist
from typing import Optional
from ..fast_utils.toeplitz import sym_toeplitz_matmul
from torch.fft import rfft, irfft


class EP_GP_circ(Rdist):
    name = "EP_GP_circ"

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

        super(EP_GP_circ, self).__init__(manif, 1)

        self.use_fast_toeplitz = use_fast_toeplitz
        self.manif = manif
        self.d = manif.d
        self.m = m

        #initialize GP mean
        nu = torch.randn((n_samples, self.d, m)) * 0.01
        self._nu = nn.Parameter(data=nu, requires_grad=True)  #m in the notes

        #initialize covariance parameters
        _scale = torch.ones(n_samples, self.d, m) * _scale  #n_diag x T
        self._scale = nn.Parameter(data=inv_softplus(_scale),
                                   requires_grad=True)

        ell = (torch.max(ts) - torch.min(ts)) / 20 if ell is None else ell
        _ell = torch.ones(1, self.d, 1) * ell
        self._ell = nn.Parameter(data=inv_softplus(_ell), requires_grad=True)

        assert m % 2 == 0
        _c = torch.ones(n_samples, self.d, int(m / 2) + 1)
        self._c = nn.Parameter(data=inv_softplus(_c), requires_grad=True)

        #pre-compute time differences
        self.ts = ts
        self.dts_sq = torch.square(ts - ts[..., :1])  #(n_samples x 1 x m)
        #sum over _input_ dimension, add an axis for _output_ dimension
        self.dts_sq = self.dts_sq.sum(-2)[:, None, ...]  #(n_samples x 1 x m)

    @property
    def scale(self) -> torch.Tensor:
        return softplus(self._scale)
    
    @property
    def c(self) -> torch.Tensor:
        return softplus(self._c)

    @property
    def nu(self) -> torch.Tensor:
        return self._nu

    @property
    def ell(self) -> torch.Tensor:
        return softplus(self._ell)

    @property
    def prms(self):
        nu = self.nu  #mean parameters

        #K^(1/2) has length scale ell/sqrt(2) if K has ell
        ell_half = self.ell / np.sqrt(2) #(1 x d x 1)

        #K^(1/2) has sig variance sigma*2^1/4*pi^(-1/4)*ell^(-1/2) if K has sigma^2
        sig_sqr_half = 1 * (2**(1 / 4)) * np.pi**(-1 / 4) * (self.ell**(-1 / 2))
        
        # (n_samples x d x m)
        K_half = sig_sqr_half * torch.exp(-self.dts_sq.to(ell_half.device) /
                                          (2 * torch.square(ell_half)))
        #(n_samples x d x m x 1)
        mu = sym_toeplitz_matmul(K_half, nu[..., None])  #mu = K_half * nu

        return mu[..., 0].transpose(-1, -2), K_half

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
        v = rand.to(mu.device)  #(n_samples x d x m x n_mc) S*v
        
        rv = rfft(v.transpose(-1, -2)) #(n_samples x d x n_mc x m/2)
        c = self.c #(n_samples x d x m/2)
        Cv = irfft( c[..., None, :] * rv ).transpose(-1, -2) #(n_samples x d x m x n_mc)
        
        #multiply by diagonal scale
        scale = self.scale  # (n_samples, d, m)
        if sample_idxs is not None:
            scale = scale[sample_idxs, ...]

        # sample from N(mu, (KS)^2)
        # Compute S @ C @ v
        SCv = scale[..., None] * Cv

        x = sym_toeplitz_matmul(K_half, SCv) #K @ S @ C @ v
        x = x.permute(-1, 0, 2, 1)  #(n_mc x n_samples x m x d)

        if batch_idxs is not None:
            mu = mu[..., batch_idxs, :]
            x = x[..., batch_idxs, :]

        x = x + mu[None, ...]  #add mean

        #(n_mc x n_samples x m x d), (n_samples x d)
        return x, lq

    def kl(self, batch_idxs=None, sample_idxs=None):
        """
        compute KL divergence analytically
        """
        #(n_samples x d x m), (n_samples x d x m)
        nu, S, c = self.nu, self.scale, self.c

        if sample_idxs is not None:
            nu = nu[sample_idxs, ...]
            S = S[sample_idxs, ...]

        Cr = irfft(self.c) #first row of C
        TrTerm = torch.square(S).sum(-1) * torch.square(Cr).sum(-1)  #(n_samples x d)
        MeanTerm = torch.square(nu).sum(-1)  #(n_samples x d)
        DimTerm = S.shape[-1]
        LogSTerm = 2 * (torch.log(S)).sum(-1)  #(n_samples x d)
        LogCTerm = 2 * (torch.log(c)).sum(-1) - torch.log(c[..., 0]) - torch.log(c[..., -1])

        kl = 0.5 * (TrTerm + MeanTerm - DimTerm - LogSTerm - LogCTerm)

        if batch_idxs is not None:
            kl = kl * len(batch_idxs) / self.m  #scale by batch size

        return kl

    def gmu_parameters(self):
        return [self.nu]

    def concentration_parameters(self):
        return [self._scale, self._ell, self._c]

    def lat_prms(self, Y=None, batch_idxs=None, sample_idxs=None):
        mu, K_half = self.prms
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
    
    def full_cov(self):
        c = self.c.detach() # (n_samples, d, m/2)
        scale = self.scale.detach() # (n_samples, d, m)
        K_half = self.prms[1].detach() # (n_samples x d x m)
        m = K_half.shape[-1]
        
        full_K_half = torch.zeros(K_half.shape[0], K_half.shape[1], m, m).to(c.device)
        for i in range(m):
            full_K_half[..., i, i:m] = K_half[..., 0:(m-i)]
            for j in range(i):
                full_K_half[..., i, i-j-1] = K_half[..., j+1]
        
        SK = torch.diag_embed(scale) @ full_K_half
        print('SK:', SK.shape)
        print('fft:', rfft(SK).shape)
        print('CSK fft:', (c[..., None, :] * rfft(SK)).shape)
        CSK = irfft(c[..., None, :] * rfft(SK)).transpose(-1, -2) #(n_samples x d x m x m)
        print('CSK:', CSK.shape)

        return CSK.transpose(-1, -2) @ CSK

    def msg(self, Y=None, batch_idxs=None, sample_idxs=None):
        mu, _ = self.lat_prms(Y=Y,
                              batch_idxs=batch_idxs,
                              sample_idxs=sample_idxs)
        scale = self.scale

        if batch_idxs is not None:
            mu = mu[:, batch_idxs, :]
            scale = scale[..., batch_idxs]
        if sample_idxs is not None:
            scale = scale[sample_idxs, ...]

        mu_mag = torch.sqrt(torch.mean(mu**2)).item()
        sig = torch.median(scale).item()

        ell = self.ell.mean().item()
        string = (' |mu| {:.3f} | sig {:.3f} | prior_ell {:.3f} |').format(
            mu_mag, sig, ell)
        return string
