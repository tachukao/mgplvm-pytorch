import abc
import torch
import torch.nn as nn
import torch.distributions as dists
from .base import Module


class Default(Module):
    name = "default"

    def __init__(self, manif):
        super().__init__()
        self.manif = manif

    @property
    def prms(self):
        return None

    def forward(self, g):
        return self.manif.lprior(g)


class Brownian(Module):
    name = "Brownian"

    def __init__(self,
                 manif,
                 kmax=5,
                 brownian_eta=None,
                 brownian_c=None,
                 fixed_brownian_eta=False,
                 fixed_brownian_c=False):
        '''
        x_t = c + w_t
        w_t = N(0, eta)
        '''
        super().__init__()
        d = manif.d
        self.d = d
        self.manif = manif
        self.kmax = kmax

        brownian_eta = torch.ones(d) if brownian_eta is None else brownian_eta
        brownian_c = torch.zeros(d) if brownian_c is None else brownian_c
        self.log_brownian_eta = nn.Parameter(
            data=torch.log(brownian_eta),
            requires_grad=(not fixed_brownian_eta))
        self.brownian_c = nn.Parameter(data=brownian_c,
                                       requires_grad=(not fixed_brownian_c))

    @property
    def prms(self):
        brownian_eta = torch.exp(self.log_brownian_eta) + 1E-16
        brownian_c = self.brownian_c
        return brownian_c, brownian_eta

    def forward(self, g):
        brownian_c, brownian_eta = self.prms
        ginv = self.manif.inverse(g)
        dg = self.manif.gmul(ginv[..., 0:-1, :], g[..., 1:, :])
        dx = self.manif.logmap(dg)
        normal = dists.Normal(loc=brownian_c, scale=torch.sqrt(brownian_eta))
        diagn = dists.Independent(normal, 1)
        return self.manif.log_q(diagn.log_prob,
                                dx,
                                self.manif.d,
                                kmax=self.kmax)
