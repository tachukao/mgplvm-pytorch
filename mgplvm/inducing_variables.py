import torch
from torch import nn, Tensor
from .base import Module
from typing import Optional


class InducingPoints(Module):

    def __init__(self,
                 n: int,
                 d: int,
                 n_z: int,
                 parameterise=None,
                 z: Optional[Tensor] = None):
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
