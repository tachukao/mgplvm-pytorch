import abc
import torch
from torch import nn, Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import transform_to, constraints
from ..utils import softplus, inv_softplus
from ..manifolds.base import Manifold
from .common import Rdist
from typing import Optional
from ..base import Module


class ReLieBase(Rdist):
    name = "ReLieBase"

    def __init__(self,
                 manif: Manifold,
                 f,
                 kmax: int = 5,
                 diagonal: bool = False):
        super(ReLieBase, self).__init__(manif, kmax)
        self.f = f
        self.diagonal = diagonal

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

    def mvn(self, gamma=None):
        gamma = self.lat_gamma() if gamma is None else gamma
        n_samples, m, _, _ = gamma.shape
        mu = torch.zeros(n_samples, m, self.d).to(gamma.device)
        return MultivariateNormal(mu, scale_tril=gamma)

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
        gmu, gamma = self.lat_prms(Y, batch_idxs, sample_idxs)
        q = self.mvn(gamma)
        # sample a batch with dims: (n_mc x n_samples x batch_size x d)
        x = q.rsample(size)
        m = x.shape[-2]
        mu = torch.zeros(m).to(gamma.device)[..., None]
        if self.diagonal:  #compute diagonal covariance
            lq = torch.stack([
                self.manif.log_q(
                    Normal(mu, gamma[..., j, j][..., None]).log_prob,
                    x[..., j, None], 1, self.kmax).sum(dim=-1)
                for j in range(self.d)
            ]).sum(dim=0)
        else:  #compute entropy with full covariance matrix
            lq = self.manif.log_q(q.log_prob, x, self.manif.d, self.kmax)

        gtilde = self.manif.expmap(x)
        # apply g_mu with dims: (n_mc x m x d)
        g = self.manif.gmul(gmu, gtilde)
        return g, lq

    def gmu_parameters(self):
        return self.f.gmu_parameters()

    def concentration_parameters(self):
        return self.f.concentration_parameters()

    def msg(self, Y=None, batch_idxs=None, sample_idxs=None):
        mu, gamma = self.lat_prms(Y=Y,
                                  batch_idxs=batch_idxs,
                                  sample_idxs=sample_idxs)
        gamma = gamma.diagonal(dim1=-1, dim2=-2)

        mu_mag = torch.sqrt(torch.mean(mu**2)).item()
        sig = torch.median(gamma).item()
        string = (' |mu| {:.3f} | sig {:.3f} |').format(mu_mag, sig)
        return string


class _F(Module):

    def __init__(self,
                 manif: Manifold,
                 m: int,
                 n_samples: int,
                 kmax: int = 5,
                 sigma: float = 1.5,
                 gamma: Optional[Tensor] = None,
                 fixed_gamma=False,
                 diagonal=False,
                 mu=None,
                 initialization: Optional[str] = 'random',
                 Y=None):

        super(_F, self).__init__()
        self.manif = manif
        self.diagonal = diagonal

        if mu is None:
            gmu = self.manif.initialize(initialization, n_samples, m, manif.d,
                                        Y)
        else:
            assert mu.shape == (n_samples, m, manif.d2)
            gmu = torch.tensor(mu)

        self.gmu = nn.Parameter(data=gmu, requires_grad=True)

        if gamma is None:
            gamma = torch.ones(n_samples, m, manif.d) * sigma
        assert gamma.shape == (n_samples, m, manif.d)

        if diagonal:
            gamma = inv_softplus(gamma)
        else:
            gamma = torch.diag_embed(gamma)
            gamma = transform_to(constraints.lower_cholesky).inv(gamma)

        assert (gamma is not None)
        self.gamma = nn.Parameter(data=gamma, requires_grad=(not fixed_gamma))

    def forward(self, Y=None, batch_idxs=None, sample_idxs=None):
        gmu, gamma = self.prms

        if sample_idxs is not None:
            gmu = gmu[sample_idxs]
            gamma = gamma[sample_idxs]

        if batch_idxs is None:
            return gmu, gamma
        else:
            return gmu[:, batch_idxs, :], gamma[:, batch_idxs, :]

    @property
    def prms(self):
        gmu = self.manif.parameterise(self.gmu)
        if self.diagonal:
            gamma = torch.diag_embed(softplus(self.gamma))
        else:
            gamma = torch.distributions.transform_to(
                MultivariateNormal.arg_constraints['scale_tril'])(self.gamma)
        return gmu, gamma

    def gmu_parameters(self):
        return [self.gmu]

    def concentration_parameters(self):
        return [self.gamma]


class ReLie(ReLieBase):
    name = "ReLie"

    def __init__(self,
                 manif: Manifold,
                 m: int,
                 n_samples: int,
                 kmax: int = 5,
                 sigma: float = 1.5,
                 gamma: Optional[Tensor] = None,
                 fixed_gamma=False,
                 diagonal=False,
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
        kmax : Optional[int]
            number of terms used in the ReLie approximation is (2kmax+1)
        sigma : Optional[float]
            initial diagonal std of the variational distribution
        intialization : Optional[str]
            string to specify type of initialization
            ('random'/'PCA'/'identity' depending on manifold)
        mu : Optional[np.ndarray]
            initialization of the vartiational means (n_samples x m x d2)
        Y : Optional[np.ndarray]
            data used to initialize latents (n x m)
            
        Notes
        -----
        gamma is the reference distribution which is inverse transformed before storing
        since it's transformed by constraints.tril when used.
        If no gammas is None, it is initialized as a diagonal matrix with value sigma
        If diagonal, constrain the covariance to be diagonal.
        The diagonal approximation is useful for T^n as it saves an exponentially growing ReLie complexity
        The diagonal approximation only works for T^n and R^n
        """

        f = _F(manif, m, n_samples, kmax, sigma, gamma, fixed_gamma, diagonal,
               mu, initialization, Y)
        super(ReLie, self).__init__(manif, f, kmax, diagonal)

    @property
    def prms(self):
        return self.f.prms
