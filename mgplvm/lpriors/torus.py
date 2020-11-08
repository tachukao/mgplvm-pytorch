import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dists
from ..manifolds import Torus
from .common import Lprior


class LpriorTorus(Lprior):
    def __init__(self, manif):
        if not isinstance(manif, Torus):
            raise Exception("VonMises prior only works with Tori manifolds")

        super().__init__(manif)


class VonMises(LpriorTorus):
    name = "VonMises"

    def __init__(self, manif, concentration=None, fixed_concentration=False):
        super().__init__(manif)
        d = manif.d
        concentration = torch.ones(
            d) if concentration is None else concentration
        self.concentration = nn.Parameter(
            data=dists.transform_to(
                dists.constraints.greater_than_eq(0)).inv(concentration),
            requires_grad=(not fixed_concentration))

    @property
    def prms(self):
        return dists.transform_to(dists.constraints.greater_than_eq(0))(
            self.concentration)

    def forward(self, g):
        concentration = self.prms
        ginv = self.manif.inverse(g)
        dg = self.manif.gmul(ginv[..., 0:-1, :], g[..., 1:, :])
        vm = dists.VonMises(loc=torch.zeros(self.d).to(g.device),
                            concentration=concentration)
        vm = dists.Independent(vm, 1)
        return vm.log_prob(dg)

    @property
    def msg(self):
        concentration = self.prms
        return ('concentration {:.3f}').format(concentration.item())


class VonMises(LpriorTorus):
    name = "VonMises"

    def __init__(self, manif, concentration=None, fixed_concentration=False):
        super().__init__(manif)
        d = manif.d
        concentration = torch.ones(
            d) if concentration is None else concentration
        self.concentration = nn.Parameter(
            data=dists.transform_to(
                dists.constraints.greater_than_eq(0)).inv(concentration),
            requires_grad=(not fixed_concentration))

    @property
    def prms(self):
        return dists.transform_to(dists.constraints.greater_than_eq(0))(
            self.concentration)

    def forward(self, g):
        concentration = self.prms
        ginv = self.manif.inverse(g)
        dg = self.manif.gmul(ginv[..., 0:-1, :], g[..., 1:, :])
        vm = dists.VonMises(loc=torch.zeros(self.d).to(g.device),
                            concentration=concentration)
        vm = dists.Independent(vm, 1)
        return vm.log_prob(dg)


class LARP(LpriorTorus):
    name = "LinkedARP"

    def __init__(self,
                 p,
                 manif,
                 mu=None,
                 phi=None,
                 concentration=None,
                 fixed_mu=False,
                 fixed_concentration=False):
        super().__init__(manif)
        d = manif.d
        self.p = p
        phi = 0.0 * torch.ones(d, p) if phi is None else phi
        self.phi = nn.Parameter(data=phi, requires_grad=True)
        self.link = lambda x: (2 * torch.atan(x)) + np.pi
        self.inv_link = lambda x: torch.tan((0.5 * x) - np.pi)
        mu = torch.zeros(d) if mu is None else mu

        self.mu = nn.Parameter(data=self.inv_link(mu),
                               requires_grad=(not fixed_mu))
        concentration = torch.ones(
            d) if concentration is None else concentration
        self.concentration = nn.Parameter(
            data=dists.transform_to(
                dists.constraints.greater_than_eq(0)).inv(concentration),
            requires_grad=(not fixed_concentration))

    @property
    def prms(self):
        concentration = dists.transform_to(
            dists.constraints.greater_than_eq(0))(self.concentration)
        mu = self.link(self.mu)
        return mu, self.phi, concentration

    def forward(self, g):
        mu, phi, concentration = self.prms
        p = self.p
        g = (g - mu) % (np.pi * 2)  # make sure it's
        delta = phi * torch.stack(
            [self.inv_link(g[..., p - j - 1:-j - 1, :]) for j in range(p)],
            axis=-1)
        muhat = mu + self.link(delta.sum(-1))
        vm = dists.VonMises(loc=muhat, concentration=concentration)
        vm = dists.Independent(vm, 1)
        return vm.log_prob(g[..., p:, :])

    @property
    def msg(self):
        mu, phi, concentration = self.prms
        return (
            'mu_avg {:.3f} | phi_avg {:.3f} | concentration {:.3f}').format(
                torch.mean(mu).item(),
                torch.mean(phi).item(), concentration.item())
