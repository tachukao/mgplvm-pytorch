import abc
import torch
import torch.nn as nn
from torch import Tensor
from ..base import Module
from typing import Optional


class Manifold(metaclass=abc.ABCMeta):

    def __init__(self, d: int):
        """
        :param d: dimensionality of the manifold
        """
        super().__init__()
        self.d = d
        self.d2 = d  # d2 = d by default

    @staticmethod
    @abc.abstractmethod
    def initialize(initialization, n_samples, m, d, Y):
        pass

    @abc.abstractmethod
    def lprior(self, g: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def gmul(self, x: Tensor, y: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def inverse(self, x: Tensor) -> Tensor:
        pass

    @staticmethod
    @abc.abstractmethod
    def parameterise(x: Tensor) -> Tensor:
        pass

    @staticmethod
    @abc.abstractmethod
    def expmap(x: Tensor) -> Tensor:
        pass

    @staticmethod
    @abc.abstractmethod
    def logmap(x: Tensor) -> Tensor:
        pass

    @staticmethod
    @abc.abstractmethod
    def log_q(p, x: Tensor, d: int, kmax: int) -> Tensor:
        pass

    @staticmethod
    @abc.abstractmethod
    def distance(x: Tensor, y: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def inducing_points(self, n: int, n_z: int, z=Optional[Tensor]):
        pass

    @abc.abstractproperty
    def name(self):
        pass
