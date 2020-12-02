import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import transform_to, constraints
from .base import Module


class MVN(Module):
    """ Parameterised zero-mean multivariate normal
    TODO: with_mu a bit hacky
    """
    name = "MVN"

    def __init__(self,
                 m,
                 d,
                 with_mu=False,
                 sigma=1.5,
                 gammas=None,
                 Tinds=None,
                 fixed_gamma=False):
        '''
        gammas is the base distribution which is inverse transformed before storing
        since it's transformed by constraints.tril when used.
        If no gammas is None, it is initialized as a diagonal matrix with value sigma
        '''
        super(MVN, self).__init__()
        self.m = m
        self.d = d
        self.with_mu = with_mu  # also learn a mean parameter

        if self.with_mu:
            self.mu = nn.Parameter(data=torch.Tensor(m, d), requires_grad=True)
            self.mu.data = torch.randn(m, d) * 0.001

        gamma = torch.diag_embed(torch.ones(m, d) * sigma)
        if gammas is not None:
            gamma[Tinds, ...] = torch.tensor(gammas,
                                             dtype=torch.get_default_dtype())

        gamma = transform_to(constraints.lower_cholesky).inv(gamma)

        if fixed_gamma:
            #don't update the covariance matrix
            self.gamma = gamma
        else:
            self.gamma = nn.Parameter(data=gamma, requires_grad=True)

    @property
    def prms(self):
        gamma = torch.distributions.transform_to(
            MultivariateNormal.arg_constraints['scale_tril'])(self.gamma)
        if self.with_mu:
            return self.mu, gamma
        return gamma

    def forward(self, batch_idxs=None):
        if self.with_mu:
            mu, gamma = self.prms
        else:
            gamma = self.prms
            mu = torch.zeros(self.m, self.d).to(gamma.device)

        if batch_idxs is not None:
            mu = mu[batch_idxs]
            gamma = gamma[batch_idxs]
        return MultivariateNormal(mu, scale_tril=gamma)
