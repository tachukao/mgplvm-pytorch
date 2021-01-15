import abc
import torch
from torch import nn, Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import transform_to, constraints
from ..utils import softplus, inv_softplus
from ..manifolds.base import Manifold
from .common import Rdist, RdistAmortized
from typing import Optional


class ReLie(Rdist):
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
        super(ReLie, self).__init__(manif, m, kmax)
        self.diagonal = diagonal

        gmudata = self.manif.initialize(initialization, self.m, self.d, Y)
        self.gmu = nn.Parameter(data=gmudata, requires_grad=True)

        gamma = torch.ones(m, self.d) * sigma
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

    def lat_gmu(self):
        gmu = self.manif.parameterise(self.gmu)
        return gmu

    def lat_gamma(self):
        if self.diagonal:
            gamma = torch.diag_embed(softplus(self.gamma))
        else:
            gamma = torch.distributions.transform_to(
                MultivariateNormal.arg_constraints['scale_tril'])(self.gamma)
        return gamma

    @property
    def prms(self):
        gmu = self.lat_gmu()
        gamma = self.lat_gamma()
        return gmu, gamma

    def lat_prms(self):
        return self.prms

    def mvn(self, gamma=None, batch_idxs=None):
        gamma = self.prms[1] if gamma is None else gamma
        mu = torch.zeros(self.m, self.d).to(gamma.device)
        if batch_idxs is not None:
            mu = mu[batch_idxs]
            gamma = gamma[batch_idxs]
        return MultivariateNormal(mu, scale_tril=gamma)

    def sample(self, size, batch_idxs=None, kmax=5):
        """
        generate samples and computes its log entropy
        """
        gmu, gamma = self.prms
        q = self.mvn(gamma, batch_idxs)
        # sample a batch with dims: (n_mc x batch_size x d)
        x = q.rsample(size)

        if self.diagonal:  #consider factorized variational distribution
            gamma = self.prms
            mu = torch.zeros(self.m).to(gamma.device)
            if batch_idxs is not None:
                gamma, mu = gamma[batch_idxs], mu[batch_idxs]
            mu = mu[..., None]
            lq = torch.stack([
                self.manif.log_q(
                    Normal(mu, gamma[..., j, j][..., None]).log_prob,
                    x[..., j, None], 1, self.kmax).sum(dim=-1)
                for j in range(self.d)
            ]).sum(dim=0)
        else:
            lq = self.manif.log_q(q.log_prob, x, self.manif.d, self.kmax)

        # transform x to group with dims (n_mc x m x d)
        gtilde = self.manif.expmap(x)

        # apply g_mu with dims: (n_mc x m x d)
        if batch_idxs is not None:
            g = self.manif.gmul(gmu[batch_idxs], gtilde)
        else:
            g = self.manif.gmul(gmu, gtilde)
        return g, lq


class ReLieAmortized(RdistAmortized):
    name = "ReLieAmortized"

    def __init__(self, manif: Manifold, f, m: int, kmax: int = 5):
        super(ReLieAmortized, self).__init__(manif, m, kmax)
        self.f = f

    def lat_prms(self, Y):
        gmu, gamma = self.f(Y)
        return gmu, gamma

    def lat_gmu(self, Y):
        return self.lat_prms(Y)[0]

    def lat_gamma(self, Y):
        return self.lat_prms(Y)[1]

    @property
    def prms(self):
        self.f.prms

    def mvn(self, gamma, batch_idxs=None):
        mu = torch.zeros(self.m, self.d).to(gamma.device)

        if batch_idxs is not None:
            mu = mu[batch_idxs]
            gamma = gamma[batch_idxs]
        return MultivariateNormal(mu, scale_tril=gamma)

    def sample(self, Y, size, kmax=5):
        """
        generate samples and computes its log entropy
        """
        gmu, gamma = self.lat_prms(Y)
        q = self.mvn(gamma, batch_idxs)
        # sample a batch with dims: (n_mc x batch_size x d)
        x = q.rsample(size)
        mu = torch.zeros(self.m).to(gamma.device)[..., None]
        lq = torch.stack([
            self.manif.log_q(
                Normal(mu, gamma[..., j, j][..., None]).log_prob,
                x[..., j, None], 1, self.kmax).sum(dim=-1)
            for j in range(self.d)
        ]).sum(dim=0)

        # apply g_mu with dims: (n_mc x m x d)
        g = self.manif.gmul(gmu[batch_idxs], gtilde)
        return g, lq
