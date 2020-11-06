import torch
from torch import Tensor
import torch.distributions
import torch.nn as nn
import abc
from .base import Module
from typing import Optional
import numpy as np

log2pi: float = np.log(2 * np.pi)


class Likelihoods(Module, metaclass=abc.ABCMeta):
    def __init__(self, n: int, m: int):
        super().__init__()
        self.n = n
        self.m = m

    @abc.abstractproperty
    def log_prob(y):
        pass

    @abc.abstractproperty
    def variational_expectation(y, mu, var):
        pass


class Gaussian(Likelihoods):
    def __init__(self, n: int, m: int, variance: Optional[Tensor] = None):
        super().__init__(n, m)
        sigma = 1 * torch.ones(n, ) if variance is None else torch.sqrt(
            torch.tensor(variance, dtype=torch.get_default_dtype()))
        self.sigma = nn.Parameter(data=sigma, requires_grad=True)

    @property
    def prms(self):
        variance = torch.square(self.sigma)
        return variance

    def log_prob(self, y):
        pass
    
    def sample(self, f_samps):
        '''f is n_b x n x m'''
        prms = self.prms
        #sample from p(y|f)
        dist = torch.distributions.Normal(f_samps, torch.sqrt(prms).reshape(1,-1,1))
        y_samps = dist.sample()
        return y_samps

    def variational_expectation(self, n_samples, y, fmu, fvar):
        n_b = fmu.shape[0]
        variance = self.prms
        ve1 = -0.5 * log2pi * self.m * self.n * n_samples * n_b
        ve2 = -0.5 * torch.log(variance).sum() * n_samples * n_b * self.m
        ve3 = -0.5 * torch.square(y - fmu) / variance[..., None, None]
        ve4 = -0.5 * fvar / variance[..., None] * n_samples
        return ve1.sum() + ve2.sum() + ve3.sum() + ve4.sum()


class Poisson(Likelihoods):
    def __init__(self,
                 n: int,
                 m: int,
                 inv_link=torch.exp,
                 binsize=1,
                 c: Optional[Tensor] = None,
                 d: Optional[Tensor] = None,
                 fixed_c=False,
                 fixed_d=False):
        super().__init__(n, m)
        self.inv_link = inv_link
        self.binsize = binsize
        c = torch.ones(n, ) if c is None else c
        d = torch.zeros(n, ) if d is None else d
        self.c = nn.Parameter(data=c, requires_grad=not fixed_c)
        self.d = nn.Parameter(data=d, requires_grad=not fixed_d)

    @property
    def prms(self):
        return self.c, self.d

    def log_prob(self, y):
        pass
    
    def sample(self, f_samps):
        c, d = self.prms
        lambd = torch.exp(c[None, ..., None] * f_samps + d[None, ..., None])
        #sample from p(y|f)
        dist = torch.distributions.Poisson(lambd)
        y_samps = dist.sample()
        return y_samps
    

    def variational_expectation(self, n_samples, y, fmu, fvar):
        if self.inv_link == torch.exp:
            c, d = self.prms
            fmu = c[..., None, None] * fmu + d[..., None, None]
            fvar = fvar * torch.square(c[..., None])
            n_b = fmu.shape[0]
            v1 = (y * fmu) - (self.binsize *
                              torch.exp(fmu + 0.5 * fvar[..., None]))
            v2 = (y * np.log(self.binsize) - torch.lgamma(y + 1)) * n_b
            return v1.sum() + v2.sum()
        else:
            raise NotImplementedError