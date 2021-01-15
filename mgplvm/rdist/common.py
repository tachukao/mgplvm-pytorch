from ..base import Module
from ..manifolds.base import Manifold


class Rdist(Module):
    def __init__(self, manif: Manifold, m: int, kmax: int):
        super(Rdist, self).__init__()
        self.manif = manif
        self.d = manif.d
        self.m = manif.m
        self.kmax = kmax
