import abc
from torch import Tensor
from ..base import Module
from ..manifolds.base import Manifold
from typing import Tuple


class Rdist(Module, metaclass=abc.ABCMeta):
    def __init__(self, manif: Manifold, kmax: int):
        super(Rdist, self).__init__()
        self.manif = manif
        self.d = manif.d
        self.kmax = kmax

    @abc.abstractmethod
    def lat_gmu(self, Y, batch_idxs, sample_idxs) -> Tensor:
        pass

    @abc.abstractmethod
    def lat_gamma(self, Y, batch_idxs, sample_idxs) -> Tensor:
        pass

    @abc.abstractmethod
    def lat_prms(self, Y, batch_idxs, sample_idxs) -> Tuple[Tensor, Tensor]:
        pass

    @abc.abstractmethod
    def sample(self, size, Y, batch_idxs, sample_idxs,
               kmax) -> Tuple[Tensor, Tensor]:
        pass

    @abc.abstractmethod
    def gmu_parameters(self):
        pass

    @abc.abstractmethod
    def concentration_parameters(self):
        pass
