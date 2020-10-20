import torch
import torch.nn as nn
from .base import Module


class InducingPoints(Module):

    def __init__(self, n, d, n_z, parameterise=None, z=None):
        super().__init__()
        self.n = n  # neurons
        self.d = d  # latent dimensionality
        self.n_z = n_z  # number of inducing points
        self.parameterise = parameterise  # project to group

        z = torch.randn(n, d, n_z) if z is None else z
        self.z = nn.Parameter(data=z, requires_grad=True)

    @property
    def prms(self):
        if self.parameterise is None:
            return self.z
        else:
            return self.parameterise(self.z)
