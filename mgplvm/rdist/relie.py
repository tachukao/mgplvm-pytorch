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

    def __init__(self, manif: Manifold, f, kmax: int = 5):
        super(ReLieBase, self).__init__(manif, kmax)
        self.f = f

    def lat_prms(self, Y=None, batch_idxs=None):
        gmu, gamma = self.f(Y, batch_idxs)
        return gmu, gamma

    def lat_gmu(self, Y=None, batch_idxs=None):
        return self.lat_prms(Y)[0]

    def lat_gamma(self, Y=None, batch_idxs=None):
        return self.lat_prms(Y, batch_idxs)[1]

    @property
    def prms(self):
        self.f.prms

    def mvn(self, gamma=None):
        gamma = self.lat_gamma() if gamma is None else gamma
        m = gamma.shape[0]
        mu = torch.zeros(m, self.d).to(gamma.device)
        return MultivariateNormal(mu, scale_tril=gamma)

    def sample(self, size, Y=None, batch_idxs=None, kmax=5):
        """
        generate samples and computes its log entropy
        """
        gmu, gamma = self.lat_prms(Y, batch_idxs)
        q = self.mvn(gamma)
        # sample a batch with dims: (n_mc x batch_size x d)
        x = q.rsample(size)
        m = x.shape[1]
        mu = torch.zeros(m).to(gamma.device)[..., None]
        lq = torch.stack([
            self.manif.log_q(
                Normal(mu, gamma[..., j, j][..., None]).log_prob,
                x[..., j, None], 1, self.kmax).sum(dim=-1)
            for j in range(self.d)
        ]).sum(dim=0)
        gtilde = self.manif.expmap(x)

        # apply g_mu with dims: (n_mc x m x d)
        g = self.manif.gmul(gmu, gtilde)
        return g, lq


class _F(Module):
    def __init__(self,
                 manif: Manifold,
                 m: int,
                 kmax: int = 5,
                 sigma: float = 1.5,
                 gammas: Optional[Tensor] = None,
                 Tinds=None,
                 fixed_gamma=False,
                 diagonal=False,
                 initialization: Optional[str] = 'random',
                 Y=None):

        super(_F, self).__init__()
        self.manif = manif
        gmudata = self.manif.initialize(initialization, m, manif.d, Y)
        self.gmu = nn.Parameter(data=gmudata, requires_grad=True)
        self.diagonal = diagonal

        gamma = torch.ones(m, manif.d) * sigma
        gamma = inv_softplus(gamma) if diagonal else torch.diag_embed(gamma)
        if gammas is not None:
            gamma[Tinds, ...] = torch.tensor(gammas,
                                             dtype=torch.get_default_dtype())

        if not diagonal:
            gamma = transform_to(constraints.lower_cholesky).inv(gamma)

        if fixed_gamma:  #don't update the covariance matrix
            self.gamma = nn.Parameter(data=gamma, requires_grad=False)
        else:
            self.gamma = nn.Parameter(data=gamma, requires_grad=True)

    def forward(self, Y=None, batch_idxs=None):
        gmu, gamma = self.prms
        if batch_idxs is None:
            return gmu, gamma
        else:
            return gmu[batch_idxs], gamma[batch_idxs]

    @property
    def prms(self):
        gmu = self.manif.parameterise(self.gmu)
        if self.diagonal:
            gamma = torch.diag_embed(softplus(self.gamma))
        else:
            gamma = torch.distributions.transform_to(
                MultivariateNormal.arg_constraints['scale_tril'])(self.gamma)
        return gmu, gamma


class ReLie(ReLieBase):
    name = "ReLie"

    def __init__(self,
                 manif: Manifold,
                 m: int,
                 kmax: int = 5,
                 sigma: float = 1.5,
                 gammas: Optional[Tensor] = None,
                 Tinds=None,
                 fixed_gamma=False,
                 diagonal=False,
                 initialization: Optional[str] = 'random',
                 Y=None):
        '''
        Notes
        -----
        gamma is the reference distribution which is inverse transformed before storing
        since it's transformed by constraints.tril when used.
        If no gammas is None, it is initialized as a diagonal matrix with value sigma
        If diagonal, constrain the covariance to be diagonal.
        The diagonal approximation is useful for T^n as it saves an exponentially growing ReLie complexity
        The diagonal approximation only works for T^n and R^n
        '''

        f = _F(manif, m, kmax, sigma, gammas, Tinds, fixed_gamma, diagonal,
               initialization, Y)
        super(ReLie, self).__init__(manif, f, kmax)

    @property
    def prms(self):
        return self.f.prms
