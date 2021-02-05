import torch
from torch import Tensor
import torch.distributions
import torch.nn as nn
import abc
from .base import Module
from typing import Optional
import torch.distributions as dists
import numpy as np
from numpy.polynomial.hermite import hermgauss
import warnings

log2pi: float = np.log(2 * np.pi)
n_gh_locs: int = 20  # default number of Gauss-Hermite points


def exp_link(x):
    '''exponential link function used for positive observations'''
    return torch.exp(x)


def id_link(x):
    '''identity link function used for neg binomial data'''
    return x


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
                 sigma: Optional[Tensor] = None,
                 n_gh_locs=n_gh_locs,
                 learn_sigma=True):
        super().__init__(n, n_gh_locs)
        sigma = 1 * torch.ones(n,) if sigma is None else sigma
        self.sigma = nn.Parameter(data=sigma, requires_grad=learn_sigma)

    @property
    def prms(self):
        variance = torch.square(self.sigma)
        return variance
    
    @property
    def sigma(self):
          return (1e-20 + self.prms).sqrt()

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

    def variational_expectation(self, y, fmu, fvar):
        """
        Parameters
        ----------
        y : Tensor
            number of MC samples (n_samples x n x m)
        fmu : Tensor
            GP mean (n_mc x n_samples x n x m)
        fvar : Tensor
            GP diagonal variance (n_mc x n_samples x n x m)

        Returns
        -------
        Log likelihood : Tensor
            SVGP likelihood term per MC, neuron, sample (n_mc x n_samples x n)
        """
        n_mc, m = fmu.shape[0], fmu.shape[-1]
        variance = self.prms  #(n)
        #print(variance.shape)
        ve1 = -0.5 * log2pi * m  #scalar
        ve2 = -0.5 * torch.log(variance) * m  #(n)
        ve3 = -0.5 * torch.square(y - fmu) / variance[
            ..., None]  #(n_mc x n_samples x n x m )
        ve4 = -0.5 * fvar / variance[..., None]  #(n_mc x n_samples x n x m)

        #(n_mc x n_samples x n)
        return ve1 + ve2 + ve3.sum(-1) + ve4.sum(-1)


class Poisson(Likelihood):

    def __init__(
            self,
            n: int,
            inv_link=exp_link,  #torch.exp,
            binsize=1,
            c: Optional[Tensor] = None,
            d: Optional[Tensor] = None,
            fixed_c=True,
            fixed_d=False,
            n_gh_locs: Optional[int] = n_gh_locs):
        super().__init__(n, n_gh_locs)
        self.inv_link = inv_link
        self.binsize = binsize
        c = torch.ones(n,) if c is None else c
        d = torch.zeros(n,) if d is None else d
        self.c = nn.Parameter(data=c, requires_grad=not fixed_c)
        self.d = nn.Parameter(data=d, requires_grad=not fixed_d)
        self.n_gh_locs = n_gh_locs

    @property
    def prms(self):
        return self.c, self.d

    def log_prob(self, lamb, y):
        #lambd: (n_mc, n_samples x n, m, n_gh)
        #y: (n, n_samples x m)
        p = dists.Poisson(lamb)
        return p.log_prob(y[None, ..., None])

    def sample(self, f_samps):
        c, d = self.prms
        lambd = self.binsize * self.inv_link(c[..., None] * f_samps +
                                             d[..., None])
        dist = torch.distributions.Poisson(lambd)
        y_samps = dist.sample()
        return y_samps

    def variational_expectation(self, y, fmu, fvar):
        """
        Parameters
        ----------
        y : Tensor
            number of MC samples (n_samples x n x m)
        fmu : Tensor
            GP mean (n_mc x n_samples x n x m)
        fvar : Tensor
            GP diagonal variance (n_mc x n_samples x n x m)

        Returns
        -------
        Log likelihood : Tensor
            SVGP likelihood term per MC, neuron, sample (n_mc x n)
        """
        c, d = self.prms
        fmu = c[..., None] * fmu + d[..., None]
        fvar = fvar * torch.square(c[..., None])
        if self.inv_link == exp_link:
            n_mc = fmu.shape[0]
            v1 = (y * fmu) - (self.binsize * torch.exp(fmu + 0.5 * fvar))
            v2 = (y * np.log(self.binsize) - torch.lgamma(y + 1))
            #v1: (n_b x n_samples x n x m)  v2: (n_samples x n x m) (per mc sample)
            lp = v1.sum(-1) + v2.sum(-1)
            return lp

        else:
            # use Gauss-Hermite quadrature to approximate integral
            locs, ws = hermgauss(self.n_gh_locs)
            ws = torch.tensor(ws, device=fmu.device)
            locs = torch.tensor(locs, device=fvar.device)
            fvar = fvar[..., None]  #add n_gh
            fmu = fmu[..., None]  #add n_gh
            locs = self.inv_link(torch.sqrt(2. * fvar) * locs +
                                 fmu) * self.binsize  #(n_mc, n, m, n_gh)
            lp = self.log_prob(locs, y)
            return 1 / np.sqrt(np.pi) * (lp * ws).sum(-1).sum(-1)
            #return torch.sum(1 / np.sqrt(np.pi) * lp * ws)


class NegativeBinomial(Likelihood):

    def __init__(self,
                 n: int,
                 inv_link=id_link,
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
        total_count = 4 * torch.ones(n,) if total_count is None else total_count
        total_count = dists.transform_to(
            dists.constraints.greater_than_eq(0)).inv(total_count)
        assert (total_count is not None)
        c = torch.ones(n,) if c is None else c
        d = torch.zeros(n,) if d is None else d
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
        '''f_samps is n_mc x n_samples x n x m'''
        total_count, c, d = self.prms
        rate = c[..., None] * f_samps + d[..., None]  #shift+scale
        rate = self.inv_link(rate) * self.binsize
        dist = dists.NegativeBinomial(total_count[None, ..., None, None],
                                      logits=rate)  #neg binom
        y_samps = dist.sample()  #sample observations
        return y_samps

    def log_prob(self, total_count, rate, y):
        #total count: (n) -> (n_mc, n_samples, n, m, n_gh)
        #rate: (n_mc, n_samples, n, m, n_gh)
        #y: (n_samples, n, m)
        p = dists.NegativeBinomial(total_count[None, ..., None, None],
                                   logits=rate)
        return p.log_prob(y[None, ..., None])

    def variational_expectation(self, y, fmu, fvar):
        """
        Parameters
        ----------
        y : Tensor
            number of MC samples (n_samples x n x m)
        fmu : Tensor
            GP mean (n_mc x n_samples x n x m)
        fvar : Tensor
            GP diagonal variance (n_mc x n_samples x n x m)

        Returns
        -------
        Log likelihood : Tensor
            SVGP likelihood term per MC, neuron, sample (n_mc x n_samples x n)
        """
        total_count, c, d = self.prms
        fmu = c[..., None] * fmu + d[..., None]
        fvar = fvar * torch.square(c[..., None])
        #print(fmu.shape, fvar.shape)
        # use Gauss-Hermite quadrature to approximate integral
        locs, ws = hermgauss(
            self.n_gh_locs)  #sample points and weights for quadrature
        ws = torch.tensor(ws, device=fmu.device)
        locs = torch.tensor(locs, device=fvar.device)
        fvar = fvar[..., None]  #add n_samples and locs
        fmu = fmu[..., None]  #add locs
        #print(locs.shape)
        locs = self.inv_link(torch.sqrt(2. * fvar) * locs +
                             fmu) * self.binsize  #coordinate transform
        #print(total_count.shape, locs.shape)
        lp = self.log_prob(total_count, locs,
                           y)  #(n_mc x n_samples x n x m, n_gh)

        #print(lp.shape, ws.shape, (lp * ws).shape)
        return 1 / np.sqrt(np.pi) * (lp * ws).sum(-1).sum(-1)
