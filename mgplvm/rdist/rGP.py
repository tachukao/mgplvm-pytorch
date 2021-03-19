import abc
import torch
import numpy as np
from torch import nn, Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import transform_to, constraints
from ..utils import softplus, inv_softplus
from ..manifolds.base import Manifold
from .common import Rdist
from typing import Optional
from ..base import Module


class GPbase(Rdist):
    name = "GPBase"

    def __init__(self, manif: Manifold, f):
        super(GPbase, self).__init__(manif, 1)
        self.f = f

    def lat_prms(self, Y=None, batch_idxs=None, sample_idxs=None):
        gmu, gamma = self.f(Y, batch_idxs, sample_idxs)
        return gmu, gamma

    def lat_gmu(self, Y=None, batch_idxs=None, sample_idxs=None):
        return self.lat_prms(Y, batch_idxs, sample_idxs)[0]

    def lat_gamma(self, Y=None, batch_idxs=None, sample_idxs=None):
        return self.lat_prms(Y, batch_idxs, sample_idxs)[1]

    @property
    def prms(self):
        self.f.prms

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
        #(n_samples x m x d), (n_samples x d x m x m)
        gmu, gamma = self.lat_prms(Y, batch_idxs, sample_idxs)
        Lq = torch.cholesky(gamma)
        q = self.mvn(Lq)
        # sample a batch with dims: (n_mc x n_samples x m x d)
        x = q.rsample(size).transpose(-1, -2)
        #print('gmu, gamma, x shapes', gmu.shape, gamma.shape, x.shape)

        if analytic_kl:
            lq = self.kl(prior, gmu.transpose(-1, -2), gamma,
                         Lq)  #(n_samples x d)
        else:
            lq = q.log_prob(x.transpose(-1, -2))  #(n_mc x n_samples x d)
            #print('lq shape:', lq.shape)
            #sum across d
            lq = lq.sum(-1)[..., None]  #(n_mc x n_samples x 1)

        #print('x, mu shapes', x.shape, gmu.shape)
        x = x + gmu[None, ...]  #add mean
        return x, lq

    def kl(self, prior, gmu, gamma_q, Lq):
        """
        compute KL divergence analytically
        gamma_q is (n_samples x d x m x m)
        gmu is (n_samples x d x m)
        """
        gamma_p = prior.prms  #(n_samples x d x m x m)
        Lp = torch.cholesky(gamma_p)
        muq = gmu
        m = Lp.shape[-1]  #dimensionality of the Gaussian
        sqTerm = torch.square(muq[..., None, :] @ torch.inverse(Lp)).sum(
            (-1, -2))  #(n_samples x d)
        logDetq = 2 * torch.log(torch.diagonal(Lq, dim1=-1, dim2=-2)).sum(
            -1)  # log |S1| (n_samples x d)
        logDetp = 2 * torch.log(torch.diagonal(Lp, dim1=-1, dim2=-2)).sum(
            -1)  # log |S2|
        cholsolve = torch.cholesky_solve(gamma_q, Lp)
        trterm = torch.diagonal(cholsolve, dim1=-1,
                                dim2=-2).sum(-1)  #(n_samples x d)
        kl = 0.5 * (trterm + sqTerm - m + logDetp - logDetq)  #(n_samples x d)

        return kl

    def gmu_parameters(self):
        return self.f.gmu_parameters()

    def concentration_parameters(self):
        return self.f.concentration_parameters()


class _F(Module):

    def __init__(self,
                 manif: Manifold,
                 m: int,
                 n_samples: int,
                 ts: Tensor,
                 mu=None,
                 initialization: Optional[str] = 'random',
                 Y=None):

        super(_F, self).__init__()
        self.manif = manif
        self.d = manif.d

        #initialize GP mean
        if mu is None:
            gmu = self.manif.initialize(initialization, n_samples, m, manif.d,
                                        Y)
        else:
            assert mu.shape == (n_samples, m, manif.d2)
            gmu = torch.tensor(mu)
        self.gmu = nn.Parameter(data=gmu, requires_grad=True)

        #initialize covariance parameters
        jitter = 1e-3
        #_scale = torch.ones(n_samples, self.d, m)*0.5
        _scale = torch.ones(1, self.d, 1) * 0.5
        ell = (torch.max(ts) - torch.min(ts)) / 10
        _ell = torch.ones(1, self.d, 1, 1) * ell

        self._scale = nn.Parameter(
            data=inv_softplus(_scale), requires_grad=True
        )  #can add separate scale for each latent in the future
        self._ell = nn.Parameter(data=inv_softplus(_ell), requires_grad=True)
        #self.noise = torch.ones(
        #    n_samples, self.d, m
        #) * jitter  #nn.Parameter(torch.ones(n_samples, self.d, m)*jitter, requires_grad=False)
        self.noise = jitter

        #pre-compute time differences
        self.dts_sq = torch.square(ts[..., None] -
                                   ts[..., None, :])  #(n_samples x 1 x m x m)
        self.dts_sq = self.dts_sq.sum(
            -3
        )[:, None,
          ...]  #sum over _input_ dimension, add an axis for _output_ dimension
        #print('dts:', self.dts_sq.shape)

    @property
    def scale(self) -> torch.Tensor:
        return softplus(self._scale)

    @property
    def ell(self) -> torch.Tensor:
        return softplus(self._ell)

    def forward(self, Y=None, batch_idxs=None, sample_idxs=None):
        gmu, gamma = self.prms

        if sample_idxs is not None:
            gmu = gmu[sample_idxs]
            gamma = gamma[sample_idxs]

        if batch_idxs is None:
            return gmu, gamma
        else:
            return gmu[:, batch_idxs, :], gamma[..., batch_idxs, :][...,
                                                                    batch_idxs]

    @property
    def prms(self):
        gmu = self.manif.parameterise(self.gmu)  #mean

        gamma = torch.exp(
            -self.dts_sq /
            (2 * torch.square(self.ell)))  #(n_samples x d x m x m)
        gamma = self.scale[..., None, :] * gamma * self.scale[..., None]
        #gamma = gamma + torch.diag_embed(torch.square(self.noise.to(gmu.device))) #covariance

        gamma = gamma + torch.eye(gamma.shape[-1]).to(
            gamma.device) * (self.noise**2)

        return gmu, gamma

    def gmu_parameters(self):
        return [self.gmu]

    def concentration_parameters(self):
        return [self._scale, self._ell]


class lat_GP(GPbase):
    name = "GP"

    def __init__(self,
                 manif: Manifold,
                 m: int,
                 n_samples: int,
                 ts: torch.Tensor,
                 mu=None,
                 initialization: Optional[str] = 'random',
                 Y=None):
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

        f = _F(manif, m, n_samples, ts, mu, initialization, Y)
        super(lat_GP, self).__init__(manif, f)

    @property
    def prms(self):
        return self.f.prms


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

        self.manif = manif
        self.d = manif.d

        #initialize GP mean
        nu = torch.randn((n_samples, self.d, m)) * 1
        self._nu = nn.Parameter(data=nu, requires_grad=True)

        #self._nu_s = nn.Parameter(data=torch.ones(1), requires_grad=True)
        #self._nu_i = nn.Parameter(data=torch.zeros(1), requires_grad=True)

        #initialize covariance parameters
        _scale = torch.ones(n_samples, self.d, m) * _scale
        self._scale = nn.Parameter(data=inv_softplus(_scale),
                                   requires_grad=True)

        ell = (torch.max(ts) - torch.min(ts)) / 20 if ell is None else ell
        _ell = torch.ones(1, self.d, 1, 1) * ell
        self._ell = nn.Parameter(data=inv_softplus(_ell), requires_grad=True)

        #pre-compute time differences #(n_samples x 1 x m x m)
        self.dts_sq = torch.square(ts[..., None] - ts[..., None, :])
        #sum over _input_ dimension, add an axis for _output_ dimension
        self.dts_sq = self.dts_sq.sum(-3)[:, None, ...]
        print('dt size:',
              self.dts_sq.element_size() * self.dts_sq.nelement() / 1e6, 'mb')

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
                                                            )  #(1 x d x 1 x 1)
        #sig_sqr_half = 1

        #(n_samples x d x m x m)
        K_half = sig_sqr_half * torch.exp(-self.dts_sq /
                                          (2 * torch.square(ell_half)))
        mu = K_half @ nu[..., None]  #(n_samples x d x m x 1)
        #multiply diagonal scale column wise to get cholesky factor
        scale = self.scale
        scale = scale / scale * scale.mean()
        K_half_S = K_half * scale[
            ..., None, :]  # (n_samples x d x m x m) * (n_samples x d x 1 x m)
        return mu[..., 0].transpose(-1, -2), K_half_S

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
        lq = self.kl(batch_idxs = batch_idxs, sample_idxs = sample_idxs)  #(n_samples x d)

        #(n_samples x m x d), (n_samples x d x m x m)
        mu, K_half_S = self.lat_prms(batch_idxs = batch_idxs, sample_idxs = sample_idxs)

        # sample a batch with dims: (n_samples x d x m x n_mc)
        x = K_half_S @ torch.randn(
            (mu.shape[0], mu.shape[2], mu.shape[1], size[0])).to(
                K_half_S.device)
        x = x.permute(-1, 0, 2, 1)  #(n_mc x n_samples x m x d)
        x = x + mu[None, ...]  #add mean

        #(n_mc x n_samples x m x d), (n_samples x d)
        return x, lq

    def kl(self, batch_idxs = None, sample_idxs = None):
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
        mu, K_half_S = self.prms
        if batch_idxs is not None:
            mu = mu[:, batch_idxs, :]
            K_half_S = K_half_S[..., :, batch_idxs][..., batch_idxs, :]
        if sample_idxs is not None:
            mu = mu[sample_idxs, ...]
            K_half_S = K_half_S[sample_idxs, ...]
            
        return mu, K_half_S

    def lat_gmu(self, Y=None, batch_idxs=None, sample_idxs=None):
        return self.lat_prms(Y = Y, batch_idxs = batch_idxs, sample_idxs = sample_idxs)[0]

    def lat_gamma(self, Y=None, batch_idxs=None, sample_idxs=None):
        return self.lat_prms(Y = Y, batch_idxs = batch_idxs, sample_idxs = sample_idxs)[1]

    def msg(self, Y=None, batch_idxs=None, sample_idxs=None):
        mu, gamma = self.lat_prms(Y=Y,
                                  batch_idxs=batch_idxs,
                                  sample_idxs=sample_idxs)
        #gamma = gamma.diagonal(dim1=-1, dim2=-2)
        #sig = torch.median(gamma).item()

        mu_mag = torch.sqrt(torch.mean(mu**2)).item()
        sig = torch.median(self.scale).item()

        ell = self.ell.mean().item()
        string = (' |mu| {:.3f} | sig {:.3f} | prior_ell {:.3f} |').format(
            mu_mag, sig, ell)
        return string
