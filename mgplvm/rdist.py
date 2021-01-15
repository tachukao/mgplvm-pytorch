import torch
from torch import nn, Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import transform_to, constraints
from .base import Module
from .manifolds.base import Manifold
from .utils import softplus, inv_softplus
from typing import Optional


class ReLie(Module):
    name = "ReLie"

    def __init__(self,
                 manif: Manifold,
                 m: int,
                 kmax: int = 5,
                 sigma: float = 1.5,
                 gammas: Optional[Tensor] = None,
                 Tinds=None,
                 fixed_gamma=False,
                 diagonal=False):
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
        super(ReLie, self).__init__()
        self.manif = manif
        self.m = m
        self.d = manif.d
        self.kmax = kmax
        d = self.d
        self.diagonal = diagonal

        gamma = torch.ones(m, d) * sigma
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

    @property
    def prms(self):
        if self.diagonal:
            gamma = torch.diag_embed(softplus(self.gamma))
        else:
            gamma = torch.distributions.transform_to(
                MultivariateNormal.arg_constraints['scale_tril'])(self.gamma)
        return gamma

    def mvn(self, batch_idxs=None):
        gamma = self.prms
        mu = torch.zeros(self.m, self.d).to(gamma.device)

        if batch_idxs is not None:
            mu = mu[batch_idxs]
            gamma = gamma[batch_idxs]
        return MultivariateNormal(mu, scale_tril=gamma)

    def sample(self, size, batch_idxs=None, kmax=5):
        """
        generate samples and computes its log entropy
        """
        q = self.mvn(batch_idxs)
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
        g = self.manif.transform(gtilde, batch_idxs=batch_idxs)
        return g, lq