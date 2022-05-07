import abc
from torch import Tensor
from ..base import Module
from ..manifolds.base import Manifold
from typing import Tuple


class LatentDistribution(Module, metaclass=abc.ABCMeta):
    """LatentDistribution base class."""

    name: str

    def __init__(self, manif: Manifold, kmax: int):
        super().__init__()
        self.manif = manif
        self.d = manif.d
        self.kmax = kmax

    @abc.abstractmethod
    def sample(self, size, Y, batch_idxs, sample_idxs, kmax, analytic_kl,
               prior) -> Tuple[Tensor, Tensor]:
        pass

    @abc.abstractmethod
    def gmu_parameters(self):
        pass

    @abc.abstractmethod
    def concentration_parameters(self):
        pass
