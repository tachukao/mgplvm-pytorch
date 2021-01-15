import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import transform_to, constraints
from .base import Module
from .utils import softplus, inv_softplus


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


class ReLie(Module):
    name = "ReLie"

    def __init__(self,
                 manif,
                 m,
                 with_mu=False,
                 kmax=5,
                 sigma=1.5,
                 gammas=None,
                 Tinds=None,
                 fixed_gamma=False,
                 diagonal=False):
        '''
        gammas is the reference distribution which is inverse transformed before storing
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
        self.with_mu = with_mu  # also learn a mean parameter
        self.diagonal = diagonal

        if self.with_mu:
            self.mu = nn.Parameter(data=torch.Tensor(m, d), requires_grad=True)
            self.mu.data = torch.randn(m, d) * 0.001

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
        if self.with_mu:
            return self.mu, gamma
        return gamma

    def mvn(self, batch_idxs=None):
        if self.with_mu:
            mu, gamma = self.prms
        else:
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
            lq = 0  #E_Q[logQ]
            for j in range(self.d):
                tril_d = gamma[..., j, j]
                q_d = Normal(mu[..., None], tril_d[..., None])
                newlq = self.manif.log_q(q_d.log_prob, x[..., j, None], 1,
                                         self.kmax)
                #print(newlq.shape)
                lq += newlq.sum(dim=-1)
        else:
            lq = self.manif.log_q(q.log_prob, x, self.manif.d, self.kmax)

        # transform x to group with dims (n_mc x m x d)
        gtilde = self.manif.expmap(x)

        # apply g_mu with dims: (n_mc x m x d)
        g = self.manif.transform(gtilde, batch_idxs=batch_idxs)
        return g, lq