import abc
from ..base import Module
from torch import Tensor


class Kernel(Module, metaclass=abc.ABCMeta):
    """
    Base kernel class
    """

    def __init__(self):
        super().__init__()

    @abc.abstractstaticmethod
    def K(self, x: Tensor, y: Tensor) -> Tensor:
        pass

    @abc.abstractstaticmethod
    def trK(self, x: Tensor) -> Tensor:
        pass

    @abc.abstractstaticmethod
    def diagK(self, x: Tensor) -> Tensor:
        pass
