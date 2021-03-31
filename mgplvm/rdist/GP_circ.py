import torch
import numpy as np
from torch import nn, Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from ..utils import softplus, inv_softplus
from ..manifolds.base import Manifold
from .GPbase import GPbase
from typing import Optional
from ..fast_utils.toeplitz import sym_toeplitz_matmul
from torch.fft import rfft, irfft


class GP_circ(GPbase):
    name = "GP_circ"

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
            
        Notes
        -----
        We parameterize our posterior as N(K2 v, K2 SCCS K2) where K2@K2 = Kprior, S is diagonal and C is circulant
        """

        super(GP_circ, self).__init__(manif,
                                      m,
                                      n_samples,
                                      ts,
                                      _scale=_scale,
                                      ell=ell)

        assert m % 2 == 0  #need to provide support for odd m

        #initialize circulant parameters
        if self.m % 2 == 0:
            _c = torch.ones(n_samples, self.d, int(m / 2) + 1)
        else:
            _c = torch.ones(n_samples, self.d, int((m + 1) / 2))
        self._c = nn.Parameter(data=inv_softplus(_c), requires_grad=True)

    @property
    def c(self) -> torch.Tensor:
        return softplus(self._c)

    @property
    def prms(self):
        return self.nu, self.scale, self.ell, self.c

    def I_v(self, v, sample_idxs=None):
        """
        Compute I @ v for some vector v.
        Here I = S C.
        v is (n_samples x d x m x n_mc) where n_samples is the number of sample_idxs
        """
        scale, c = self.scale, self.c
        if sample_idxs is not None:
            scale = scale[sample_idxs, ...]  #(n_samples x d x m)
            c = c[sample_idxs, ...]  #(n_samples x d x m/2)

        #Fourier transform (n_samples x d x n_mc x m/2)
        rv = rfft(v.transpose(-1, -2).to(scale.device))
        #inverse fourier transform of product (n_samples x d x m x n_mc)
        Cv = irfft(c[..., None, :] * rv).transpose(-1, -2)

        #multiply by diagonal scale
        SCv = scale[..., None] * Cv

        return SCv

    def kl(self, batch_idxs=None, sample_idxs=None):
        """
        Compute KL divergence between prior and posterior.
        This should be implemented for each class separately
        """
        #(n_samples x d x m), (n_samples x d x m), (n_samples x d x m/2)
        nu, S, c = self.nu, self.scale, self.c

        if sample_idxs is not None:
            nu = nu[sample_idxs, ...]
            S = S[sample_idxs, ...]
            c = c[sample_idxs, ...]

        Cr = irfft(self.c)  #first row of C given by inverse Fourier transform
        TrTerm = torch.square(S).sum(-1) * torch.square(Cr).sum(
            -1)  #(n_samples x d)
        MeanTerm = torch.square(nu).sum(-1)  #(n_samples x d)
        DimTerm = S.shape[-1]
        LogSTerm = 2 * (torch.log(S)).sum(-1)  #(n_samples x d)

        #c[0] + 2*c[1:end] (n_samples x d)
        LogCTerm = 2 * (torch.log(c)).sum(-1) - torch.log(c[..., 0])
        if self.m % 2 == 0:
            #c[0] + c[-1] + 2*c[1:-1]
            LogCTerm = LogCTerm - torch.log(c[..., -1])

        kl = 0.5 * (TrTerm + MeanTerm - DimTerm - LogSTerm - LogCTerm)
        if batch_idxs is not None:  #scale by batch size
            kl = kl * len(batch_idxs) / self.m

        return kl

    def full_cov(self):
        """there is definitely a better way of doing this"""
        c = self.c.detach()  # (n_samples, d, m/2)
        scale = self.scale.detach()  # (n_samples, d, m)
        K_half = self.prms[1].detach()  # (n_samples x d x m)
        m = K_half.shape[-1]

        full_K_half = torch.zeros(K_half.shape[0], K_half.shape[1], m,
                                  m).to(c.device)
        for i in range(m):
            full_K_half[..., i, i:m] = K_half[..., 0:(m - i)]
            for j in range(i):
                full_K_half[..., i, i - j - 1] = K_half[..., j + 1]

        SK = torch.diag_embed(scale) @ full_K_half
        print('SK:', SK.shape)
        print('fft:', rfft(SK).shape)
        print('CSK fft:', (c[..., None, :] * rfft(SK)).shape)
        CSK = irfft(c[..., None, :] * rfft(SK)).transpose(
            -1, -2)  #(n_samples x d x m x m)
        print('CSK:', CSK.shape)

        return CSK.transpose(-1, -2) @ CSK
