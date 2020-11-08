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
