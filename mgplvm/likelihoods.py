import torch
from torch import Tensor
import torch.distributions
import torch.nn as nn
import abc
from .base import Module
from typing import Optional
import torch.distributions as dists
import numpy as np

log2pi: float = np.log(2 * np.pi)
n_gh_locs: int = 20  # default number of Gauss-Hermite points


class Likelihood(Module, metaclass=abc.ABCMeta):
    def __init__(self, n: int, n_gh_locs: int):
        super().__init__()
        self.n = n
        self.n_gh_locs = n_gh_locs

    @abc.abstractproperty
    def log_prob(y):
        pass

    @abc.abstractproperty
    def variational_expectation(y, mu, var):
        pass


class Gaussian(Likelihood):
    def __init__(self,
                 n: int,
                 variance: Optional[Tensor] = None,
                 n_gh_locs=n_gh_locs):
        super().__init__(n, n_gh_locs)
        sigma = 1 * torch.ones(n, ) if variance is None else torch.sqrt(
            torch.tensor(variance, dtype=torch.get_default_dtype()))
        self.sigma = nn.Parameter(data=sigma, requires_grad=True)

    @property
    def prms(self):
        variance = torch.square(self.sigma)
        return variance

    def log_prob(self, y):
        raise Exception("Gaussian likelihood not implemented")

    def sample(self, f_samps):
        '''f is n_b x n x m'''
        prms = self.prms
        #sample from p(y|f)
        dist = torch.distributions.Normal(f_samps,
                                          torch.sqrt(prms).reshape(1, -1, 1))
        y_samps = dist.sample()
        return y_samps

    def variational_expectation(self, n_samples, y, fmu, fvar):
        n_b, m = fmu.shape[0], fmu.shape[2]
        variance = self.prms
        ve1 = -0.5 * log2pi * m * self.n * n_samples * n_b
        ve2 = -0.5 * torch.log(variance).sum() * n_samples * n_b
        ve3 = -0.5 * torch.square(y - fmu) / variance[..., None, None]
        ve4 = -0.5 * fvar / variance[..., None] * n_samples
        return ve1.sum() + ve2.sum() + ve3.sum() + ve4.sum()


class Poisson(Likelihood):
    def __init__(self,
                 n: int,
                 inv_link=torch.exp,
                 binsize=1,
                 c: Optional[Tensor] = None,
                 d: Optional[Tensor] = None,
                 fixed_c=False,
                 fixed_d=False,
                 n_gh_locs: Optional[int] = n_gh_locs):
        super().__init__(n, n_gh_locs)
        self.inv_link = inv_link
        self.binsize = binsize
        c = torch.ones(n, ) if c is None else c
        d = torch.zeros(n, ) if d is None else d
        self.c = nn.Parameter(data=c, requires_grad=not fixed_c)
        self.d = nn.Parameter(data=d, requires_grad=not fixed_d)

    @property
    def prms(self):
        return self.c, self.d

    def log_prob(self, lamb, y):
        if y.shape[-1] == 1:
            p = dists.Poisson(lamb)
            return p.log_prob(y)
        else:
            p = dists.Poisson(lamb[..., None, :])
            return p.log_prob(y[..., None])

    def sample(self, f_samps):
        c, d = self.prms
        lambd = self.binsize * self.inv_link(c[None, ..., None] * f_samps +
                                             d[None, ..., None])
        #sample from p(y|f)
        dist = torch.distributions.Poisson(lambd)
        y_samps = dist.sample()
        return y_samps

    def variational_expectation(self, n_samples, y, fmu, fvar, gh=False):
        c, d = self.prms
        fmu = c[..., None, None] * fmu + d[..., None, None]
        fvar = fvar * torch.square(c[..., None])
        if self.inv_link == torch.exp and (not gh):
            n_b = fmu.shape[0]
            v1 = (y * fmu) - (self.binsize *
                              torch.exp(fmu + 0.5 * fvar[..., None]))
            v2 = (y * np.log(self.binsize) - torch.lgamma(y + 1)) * n_b
            return v1.sum() + v2.sum()
        else:
            # use Gauss-Hermite quadrature to approximate integral
            locs, ws = np.polynomial.hermite.hermgauss(self.n_gh_locs)
            ws = torch.Tensor(ws).to(fmu.device)
            locs = torch.Tensor(locs).to(fvar.device)
            fvar = fvar[..., None]
            locs = self.inv_link(torch.sqrt(2. * fvar) * locs +
                                 fmu) * self.binsize
            lp = self.log_prob(locs, y)
            return torch.sum(1 / np.sqrt(np.pi) * lp * ws)


class NegativeBinomial(Likelihood):
    def __init__(self,
                 n: int,
                 inv_link=lambda x: x,
                 binsize=1,
                 total_count: Optional[Tensor] = None,
                 c: Optional[Tensor] = None,
                 d: Optional[Tensor] = None,
                 fixed_total_count=False,
                 fixed_c=False,
                 fixed_d=False,
                 n_gh_locs: Optional[int] = n_gh_locs):
        super().__init__(n, n_gh_locs)
        self.inv_link = inv_link
        self.binsize = binsize
        total_count = 1000 * torch.ones(
            n, ) if total_count is None else total_count
        total_count = dists.transform_to(
            dists.constraints.greater_than_eq(0)).inv(total_count)
        c = torch.ones(n, ) if c is None else c
        d = torch.zeros(n, ) if d is None else d
        self.total_count = nn.Parameter(data=total_count,
                                        requires_grad=not fixed_total_count)
        self.c = nn.Parameter(data=c, requires_grad=not fixed_c)
        self.d = nn.Parameter(data=d, requires_grad=not fixed_d)

    @property
    def prms(self):
        total_count = dists.transform_to(dists.constraints.greater_than_eq(0))(
            self.total_count)
        return total_count, self.c, self.d

    def sample(self, f_samps):
        '''f_samps is n_b x n x m'''
        total_count, c, d = self.prms
        rate = c[None, ..., None] * f_samps + d[None, ..., None]  #shift+scale
        rate = self.inv_link(rate) * self.binsize
        dist = dists.NegativeBinomial(total_count[None, ..., None],
                                      logits=rate)  #neg binom
        y_samps = dist.sample()  #sample observations
        return y_samps

    def log_prob(self, total_count, rate, y):
        if y.shape[-1] == 1:
            p = dists.NegativeBinomial(total_count[..., None, None],
                                       logits=rate)
            return p.log_prob(y)
        else:
            p = dists.NegativeBinomial(total_count[..., None, None, None],
                                       logits=rate[..., None, :])
            return p.log_prob(y[..., None])

    def variational_expectation(self, n_samples, y, fmu, fvar):
        total_count, c, d = self.prms
        fmu = c[..., None, None] * fmu + d[..., None, None]
        fvar = fvar * torch.square(c[..., None])
        # use Gauss-Hermite quadrature to approximate integral
        locs, ws = np.polynomial.hermite.hermgauss(
            self.n_gh_locs)  #sample points and weights for quadrature
        ws = torch.Tensor(ws).to(fmu.device)
        locs = torch.Tensor(locs).to(fvar.device)
        fvar = fvar[..., None]
        locs = self.inv_link(torch.sqrt(2. * fvar) * locs +
                             fmu) * self.binsize  #coordinate transform
        lp = self.log_prob(total_count, locs, y)
        return torch.sum(1 / np.sqrt(np.pi) * lp * ws)


class CMPoisson(Likelihood):
    """
    Conway-Maxwell-Poisson
    """
    def __init__(self,
                 n: int,
                 inv_link=torch.exp,
                 binsize=1,
                 nu: Optional[Tensor] = None,
                 c: Optional[Tensor] = None,
                 d: Optional[Tensor] = None,
                 fixed_nu=False,
                 fixed_c=False,
                 fixed_d=False,
                 n_gh_locs: Optional[int] = n_gh_locs,
                 max_k=None):
        super().__init__(n, n_gh_locs)
        self.inv_link = inv_link
        self.binsize = binsize
        self.max_k = max_k
        nu = torch.ones(n, ) if nu is None else nu
        nu = dists.transform_to(dists.constraints.greater_than_eq(0)).inv(nu)
        c = torch.ones(n, ) if c is None else c
        d = torch.zeros(n, ) if d is None else d
        self.nu = nn.Parameter(data=nu, requires_grad=not fixed_nu)
        self.c = nn.Parameter(data=c, requires_grad=not fixed_c)
        self.d = nn.Parameter(data=d, requires_grad=not fixed_d)

    @property
    def prms(self):
        nu = dists.transform_to(dists.constraints.greater_than_eq(0))(self.nu)
        return nu, self.c, self.d

    def sample(self):
        raise Exception("CMP sampling not implemented!")

    def log_prob(self, nu, rate, y):
        # we do not want to differentiate through this
        max_k = torch.max(y).item()
        K = np.max((1000, max_k * 2))
        K = max_k
        if y.shape[-1] == 1:
            nu = nu[..., None, None]
            p = y * torch.log(rate) - (nu * torch.lgamma(y + 1))
            ks = torch.arange(0, K + 1)
        else:
            nu = nu[..., None, None, None]
            y = y[..., None]
            rate = rate[..., None, :]
            p = y * torch.log(rate) - (nu * torch.lgamma(y + 1))
        ks = torch.arange(0, K + 1).to(y)
        # normalizing constant
        M = torch.logsumexp(ks * torch.log(rate[..., None]) -
                            (nu[..., None] * torch.lgamma(ks + 1)),
                            axis=-1)
        lp = p - M
        return lp

    def variational_expectation(self, n_samples, y, fmu, fvar):
        # we compute a lower bound here to the likelihood
        nu, c, d = self.prms
        fmu = c[..., None, None] * fmu + d[..., None, None]
        fvar = fvar * torch.square(c[..., None])

        if self.inv_link == torch.exp:
            # here we use a lower-bound to the variational expections
            # this is much more memory-efficient than GH quadrature approximation
            n_b = fmu.shape[0]

            K = torch.max(y).item() if self.max_k is None else self.max_k
            nu = nu[..., None, None]
            p = y * fmu + (y * np.log(self.binsize)) - (nu *
                                                           torch.lgamma(y + 1))
            ks = torch.arange(1, K + 1).to(y)
            # normalizing constant
            M = torch.logsumexp(
                (ks * fmu[..., None]) + (ks * np.log(self.binsize)) -
                (nu[..., None] * torch.lgamma(ks + 1)) +
                (0.5 * fvar[..., None, None] * torch.square(ks)),
                axis=-1)
            lp = p - M
            return lp
        else:
            # use Gauss-Hermite quadrature to approximate integral
            locs, ws = np.polynomial.hermite.hermgauss(
                self.n_gh_locs)  #sample points and weights for quadrature
            ws = torch.Tensor(ws).to(fmu.device)
            locs = torch.Tensor(locs).to(fvar.device)
            fvar = fvar[..., None]
            locs = self.inv_link(torch.sqrt(2. * fvar) * locs +
                                 fmu) * self.binsize  #coordinate transform
            lp = self.log_prob(nu, locs, y)
            return torch.sum(1 / np.sqrt(np.pi) * lp * ws)
