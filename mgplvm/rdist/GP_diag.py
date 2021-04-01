import torch
import numpy as np
from torch import nn, Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from ..utils import softplus, inv_softplus
from ..manifolds.base import Manifold
from .GPbase import GPbase
from typing import Optional
from ..fast_utils.toeplitz import sym_toeplitz_matmul


class GP_diag(GPbase):
    name = "GP_diag"

    def __init__(
        self,
        manif: Manifold,
        m: int,
        n_samples: int,
        ts: torch.Tensor,
        _scale=0.9,
        ell=None,
    ):
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
        We parameterize our posterior as N(K2 v, K2 S^2 K2) where K2@K2 = Kprior and S is diagonal
        """
        super(GP_diag, self).__init__(manif,
                                      m,
                                      n_samples,
                                      ts,
                                      _scale=_scale,
                                      ell=ell)

    def I_v(self, v, sample_idxs=None):
        """
        Compute I @ v for some vector v.
        Here I = S = diag(scale).
        v is (n_samples x d x m x n_mc) where n_samples is the number of sample_idxs
        """
        scale = self.scale  # (n_samples, d, m)
        if sample_idxs is not None:
            scale = scale[sample_idxs, ...]
        #S @ v (n_samples x d x m x n_mc)
        Sv = scale[..., None] * v.to(scale.device)
        return Sv

    def kl(self, batch_idxs=None, sample_idxs=None):
        """
        Compute KL divergence between prior and posterior.
        This should be implemented for each class separately
        """
        #(n_samples x d x m), (n_samples x d x m)
        nu, S = self.nu, self.scale

        if sample_idxs is not None:
            nu = nu[sample_idxs, ...]
            S = S[sample_idxs, ...]

        TrTerm = torch.square(S).sum(-1)  #(n_samples x d)
        MeanTerm = torch.square(nu).sum(-1)  #(n_samples x d)
        DimTerm = S.shape[-1]
        LogTerm = 2 * (torch.log(S)).sum(-1)  #(n_samples x d)

        kl = 0.5 * (TrTerm + MeanTerm - DimTerm - LogTerm)

        if batch_idxs is not None:
            kl = kl * len(batch_idxs) / self.m  #scale by batch size

        return kl
