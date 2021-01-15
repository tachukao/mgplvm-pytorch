import abc
from torch import Tensor
from ..base import Module
from ..manifolds.base import Manifold
from typing import Tuple


class Rdist(Module, metaclass=abc.ABCMeta):
    def __init__(self, manif: Manifold, m: int, kmax: int):
        super(Rdist, self).__init__()
        self.manif = manif
        self.d = manif.d
        self.m = manif.m
        self.kmax = kmax

    @abc.abstractmethod
    def lat_gmu(self) -> Tensor:
        pass

    @abc.abstractmethod
    def lat_gamma(self) -> Tensor:
        pass

    @abc.abstractmethod
    def lat_prms(self) -> Tuple[Tensor, Tensor]:
        pass


class RdistAmortized(Module, metaclass=abc.ABCMeta):
    def __init__(self, manif: Manifold, m: int, kmax: int):
        super(Rdist, self).__init__()
        self.manif = manif
        self.d = manif.d
        self.m = manif.m
        self.kmax = kmax

    @abc.abstractmethod
    def lat_gmu(self, Y) -> Tensor:
        pass

    @abc.abstractmethod
    def lat_gamma(self, Y) -> Tensor:
        pass

    @abc.abstractmethod
    def lat_prms(self, Y) -> Tuple[Tensor, Tensor]:
        pass
