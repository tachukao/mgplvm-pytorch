import abc
import torch
import torch.nn as nn
from torch import Tensor
from ..base import Module
from typing import Any


class Manifold(Module, metaclass=abc.ABCMeta):

    def __init__(self, d: int):
        """
        :param d: dimensionality of the manifold
        """
        super().__init__()
        self.d = d

    @abc.abstractmethod
    def expmap(x: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def logmap(x: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def log_q(x: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def distance(x: Tensor, y: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def inducing_points(n):
        pass
