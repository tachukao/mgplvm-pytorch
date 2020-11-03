import torch
from torch import Tensor
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
                 variance: Optional[Tensor] = None,
                 inv_link=torch.exp):
        super().__init__(n, m)
        sigma = torch.randn(n, ) if variance is None else torch.sqrt(
            torch.tensor(variance, dtype=torch.get_default_dtype()))
        self.sigma = nn.Parameter(data=sigma, requires_grad=True)
        self.inv_lin = inv_link

    @property
    def prms(self):
        variance = torch.square(self.sigma)
        return variance

    def log_prob(self, y):
        pass

    def variational_expectation(self, n_samples, y, mu, var):
        if self.inv_link == torch.exp:
            variance = self.prms
            n_b = mu.shape[0]
            v1 = y * mu - torch.exp(mu + var / 2)
            v2 = (torch.lgamma(y + 1) + y) * n_b
            return v1.sum() + v2.sum()
        else:
            raise NotImplementedError