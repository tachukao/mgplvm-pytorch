import torch
from torch import Tensor
import torch.distributions
import torch.nn as nn
import abc
from .base import Module
from typing import Optional
import torch.distributions as dists
import numpy as np
import warnings

log2pi: float = np.log(2 * np.pi)
n_gh_locs: int = 20  # default number of Gauss-Hermite points


class Likelihood(Module, metaclass=abc.ABCMeta):
    def __init__(self, n: int, n_gh_locs: Optional[int] = n_gh_locs):
        super().__init__()
        self.n = n
        self.n_gh_locs = n_gh_locs

    @abc.abstractproperty
    def log_prob(self):
        pass

    @abc.abstractproperty
    def variational_expectation(self):
        pass

    @abc.abstractstaticmethod
    def sample(self, x: Tensor):
        pass


class Gaussian(Likelihood):
    def __init__(self,
                 n: int,
                 variance: Optional[Tensor] = None,
                 n_gh_locs=n_gh_locs,
                learn_sigma = True):
        super().__init__(n, n_gh_locs)
        sigma = 1 * torch.ones(n, ) if variance is None else torch.sqrt(
            torch.tensor(variance, dtype=torch.get_default_dtype()))
        
        if learn_sigma:
            self.sigma = nn.Parameter(data=sigma, requires_grad=True)
        else:
            self.sigma = nn.Parameter(data=sigma, requires_grad=False)

    @property
    def prms(self):
        variance = torch.square(self.sigma)
        return variance

    def log_prob(self, y):
        raise Exception("Gaussian likelihood not implemented")

    def sample(self, f_samps: Tensor) -> Tensor:
        '''f is n_b x n x m'''
        prms = self.prms
        #sample from p(y|f)
        dist = torch.distributions.Normal(f_samps,
                                          torch.sqrt(prms).reshape(1, -1, 1))
        y_samps = dist.sample()
        return y_samps

    def variational_expectation(self, n_samples, y, fmu, fvar):
        """
        Parameters
        ----------
        n_samples : int
            number of samples
        y : Tensor
            number of MC samples (n x m x n_samples)
        f_mu : Tensor
            GP mean (n_mc x n x m x n_samples)
        f_var : Tensor
            GP diagonal variance (n_mc x n x m)

        Returns
        -------
        Log likelihood : Tensor
            SVGP likelihood term per MC, neuron, sample (n_mc x n x n_samples)
        """
        n_mc, m = fmu.shape[0], fmu.shape[2]
        variance = self.prms #(n)
        #print(variance.shape)
        ve1 = -0.5 * log2pi * m #scalar
        ve2 = -0.5 * torch.log(variance) * m #(n)
        ve3 = -0.5 * torch.square(y - fmu) / variance[..., None, None] #(n_mc x n x m x n_samples)
        ve4 = -0.5 * fvar / variance[..., None] #(n_mc x n x m)
            
        #print(ve1.shape, ve2.shape, ve3.shape, ve4.shape)
        exp = ve1 + ve2[None, ..., None] + ve3.sum(-2) + ve4.sum(-1)[..., None]
        #(n_mc x n x n_samples)
        #print(exp.shape)
        '''
        else:
            exp = ve1.sum() + ve2.sum() + ve3.sum() + ve4.sum()
        '''
        return exp


class Poisson(Likelihood):
    def __init__(self,
                 n: int,
                 inv_link='exp',#torch.exp,
                 binsize=1,
                 c: Optional[Tensor] = None,
                 d: Optional[Tensor] = None,
                 fixed_c=True,
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
        if False:#y.shape[-1] == 1:
            p = dists.Poisson(lamb)
            return p.log_prob(y)
        else:
            #lambd: (n_mc, n, m, n_samples, n_gh)
            #y: (n, m, n_samples)
            p = dists.Poisson(lamb)
            return p.log_prob(y[None, ..., None])

    def sample(self, f_samps):
        c, d = self.prms
        
        #######
        if self.inv_link == 'exp':
            lambd = self.binsize * torch.exp(c[None, ..., None] * f_samps +
                                             d[None, ..., None])
        else:    
            lambd = self.binsize * self.inv_link(c[None, ..., None] * f_samps +
                                             d[None, ..., None])
        #######
            
        #lambd = self.binsize * self.inv_link(c[None, ..., None] * f_samps +
        #                                     d[None, ..., None])
        #sample from p(y|f)
        dist = torch.distributions.Poisson(lambd)
        y_samps = dist.sample()
        return y_samps

    def variational_expectation(self, n_samples, y, fmu, fvar, gh=False,
                               by_sample = False):
        """
        Parameters
        ----------
        n_samples : int
            number of samples
        y : Tensor
            number of MC samples (n x m x n_samples)
        f_mu : Tensor
            GP mean (n_mc x n x m x n_samples)
        f_var : Tensor
            GP diagonal variance (n_mc x n x m)
        gh [optional] : int
            number of points used for Gauss-Hermite quadrature

        Returns
        -------
        Log likelihood : Tensor
            SVGP likelihood term per MC, neuron, sample (n_mc x n x n_samples)
        """
        c, d = self.prms
        fmu = c[..., None, None] * fmu + d[..., None, None]
        fvar = fvar * torch.square(c[..., None])
        #if self.inv_link == torch.exp and (not gh):
        if self.inv_link == 'exp' and (not gh):
            n_mc = fmu.shape[0]
            v1 = (y * fmu) - (self.binsize *
                              torch.exp(fmu + 0.5 * fvar[..., None]))
            v2 = (y * np.log(self.binsize) - torch.lgamma(y + 1))
            #v1: (n_b x n x m x n_samples)  v2: (n x m x n_samples) (per mc sample)
            return v1.sum(-2) + v2.sum(-2)[None, ...]

        else:
            # use Gauss-Hermite quadrature to approximate integral
            locs, ws = np.polynomial.hermite.hermgauss(self.n_gh_locs)
            ws = torch.Tensor(ws).to(fmu.device)
            locs = torch.Tensor(locs).to(fvar.device)
            fvar = fvar[..., None, None] #add n_samples, n_gh
            fmu = fmu[..., None] #add n_gh
            locs = self.inv_link(torch.sqrt(2. * fvar) * locs +
                                 fmu) * self.binsize #(n_mc, n, m, n_samples, n_gh)
            lp = self.log_prob(locs, y)
            return 1/np.sqrt(np.pi) * (lp * ws).sum(-1).sum(-2)
            #return torch.sum(1 / np.sqrt(np.pi) * lp * ws)


class NegativeBinomial(Likelihood):
    def __init__(self,
                 n: int,
                 inv_link=lambda x: x,
                 binsize=1,
                 total_count: Optional[Tensor] = None,
                 c: Optional[Tensor] = None,
                 d: Optional[Tensor] = None,
                 fixed_total_count=False,
                 fixed_c=True,
                 fixed_d=False,
                 n_gh_locs: Optional[int] = n_gh_locs):
        super().__init__(n, n_gh_locs)
        self.inv_link = inv_link
        self.binsize = binsize
        total_count = 4 * torch.ones(
            n, ) if total_count is None else total_count
        total_count = dists.transform_to(
            dists.constraints.greater_than_eq(0)).inv(total_count)
        assert (total_count is not None)
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
        if False:#y.shape[-1] == 1:
            p = dists.NegativeBinomial(total_count[..., None, None],
                                       logits=rate)
            return p.log_prob(y)
        else:
            #total count: (n) -> (n_mc, n, m, n_samples, n_gh)
            #rate: (n_mc, n, m, n_samples, n_gh)
            #y: (n, m, n_samples)
            p = dists.NegativeBinomial(total_count[None, ..., None, None, None],
                                       logits=rate)
            return p.log_prob(y[None, ..., None])

    def variational_expectation(self, n_samples, y, fmu, fvar, by_sample = False):
        """
        Parameters
        ----------
        n_samples : int
            number of samples
        y : Tensor
            number of MC samples (n x m x n_samples)
        f_mu : Tensor
            GP mean (n_mc x n x m x n_samples)
        f_var : Tensor
            GP diagonal variance (n_mc x n x m)

        Returns
        -------
        Log likelihood : Tensor
            SVGP likelihood term per MC, neuron, sample (n_mc x n x n_samples)
        """
        total_count, c, d = self.prms
        fmu = c[..., None, None] * fmu + d[..., None, None]
        fvar = fvar * torch.square(c[..., None])
        #print(fmu.shape, fvar.shape)
        # use Gauss-Hermite quadrature to approximate integral
        locs, ws = np.polynomial.hermite.hermgauss(
            self.n_gh_locs)  #sample points and weights for quadrature
        ws = torch.Tensor(ws).to(fmu.device)
        locs = torch.Tensor(locs).to(fvar.device)
        fvar = fvar[..., None, None] #add n_samples and locs
        fmu = fmu[..., None] #add locs
        #print(locs.shape)
        locs = self.inv_link(torch.sqrt(2. * fvar) * locs +
                             fmu) * self.binsize  #coordinate transform
        #print(total_count.shape, locs.shape)
        lp = self.log_prob(total_count, locs, y) #(n_mc x n x m x n_samples, n_gh)

        #print(lp.shape, ws.shape, (lp * ws).shape)
        return 1/np.sqrt(np.pi) * (lp * ws).sum(-1).sum(-2)
        
        #else:
        #    return torch.sum(1 / np.sqrt(np.pi) * lp * ws)        


# class CMPoisson(Likelihood):
#     """
#     Conway-Maxwell-Poisson
#     """
#     def __init__(self,
#                  n: int,
#                  inv_link=torch.exp,
#                  binsize=1,
#                  nu: Optional[Tensor] = None,
#                  c: Optional[Tensor] = None,
#                  d: Optional[Tensor] = None,
#                  fixed_nu=False,
#                  fixed_c=False,
#                  fixed_d=False,
#                  n_gh_locs: Optional[int] = n_gh_locs,
#                  max_k=None):
#         super().__init__(n, n_gh_locs)
#         self.inv_link = inv_link
#         self.binsize = binsize
#         self.max_k = max_k
#         nu = torch.ones(n, ) if nu is None else nu
#         nu = dists.transform_to(dists.constraints.greater_than_eq(0)).inv(nu)
#         assert (nu is not None)
#         c = torch.ones(n, ) if c is None else c
#         d = torch.zeros(n, ) if d is None else d
#         self.nu = nn.Parameter(data=nu, requires_grad=not fixed_nu)
#         self.c = nn.Parameter(data=c, requires_grad=not fixed_c)
#         self.d = nn.Parameter(data=d, requires_grad=not fixed_d)

#     @property
#     def prms(self):
#         nu = dists.transform_to(dists.constraints.greater_than_eq(0))(self.nu)
#         return nu, self.c, self.d

#     def sample(self):
#         raise Exception("CMP sampling not implemented!")

#     def log_prob(self, nu, rate, y):
#         # we do not want to differentiate through this
#         max_k = torch.max(y).item()
#         K = np.max((1000, max_k * 2))
#         K = max_k
#         if y.shape[-1] == 1:
#             nu = nu[..., None, None]
#             p = y * torch.log(rate) - (nu * torch.lgamma(y + 1))
#             ks = torch.arange(0, K + 1)
#         else:
#             nu = nu[..., None, None, None]
#             y = y[..., None]
#             rate = rate[..., None, :]
#             p = y * torch.log(rate) - (nu * torch.lgamma(y + 1))
#         ks = torch.arange(0, K + 1).to(y)
#         # normalizing constant
#         M = torch.logsumexp(ks * torch.log(rate[..., None]) -
#                             (nu[..., None] * torch.lgamma(ks + 1)),
#                             dim=-1)
#         lp = p - M
#         return lp

#     def variational_expectation(self, n_samples, y, fmu, fvar, by_batch=False, by_sample = False):
#         warnings.warn(
#             "CMP variational expectation not properly tested: use with caution"
#         )
#         nu, c, d = self.prms
#         fmu = c[..., None, None] * fmu + d[..., None, None]
#         fvar = fvar * torch.square(c[..., None])

#         if self.inv_link == torch.exp:
#             # here we use a lower-bound to the variational expections
#             # this is much more memory-efficient than GH quadrature approximation
#             n_b = fmu.shape[0]

#             K = torch.max(y).item() if self.max_k is None else self.max_k
#             nu = nu[..., None, None]
#             p = y * fmu + (y * np.log(self.binsize)) - (nu *
#                                                         torch.lgamma(y + 1))
#             ks = torch.arange(1, K + 1).to(y)
#             # normalizing constant
#             M = torch.logsumexp(
#                 (ks * fmu[..., None]) + (ks * np.log(self.binsize)) -
#                 (nu[..., None] * torch.lgamma(ks + 1)) +
#                 (0.5 * fvar[..., None, None] * torch.square(ks)),
#                 dim=-1)
#             lp = p - M
#             return lp
#         else:
#             # use Gauss-Hermite quadrature to approximate integral
#             locs, ws = np.polynomial.hermite.hermgauss(
#                 self.n_gh_locs)  #sample points and weights for quadrature
#             ws = torch.Tensor(ws).to(fmu.device)
#             locs = torch.Tensor(locs).to(fvar.device)
#             fvar = fvar[..., None]
#             locs = self.inv_link(torch.sqrt(2. * fvar) * locs +
#                                  fmu) * self.binsize  #coordinate transform
#             lp = self.log_prob(nu, locs, y)
#             return torch.sum(1 / np.sqrt(np.pi) * lp * ws)
