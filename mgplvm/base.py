import abc
import torch
import torch.nn as nn
from typing import Tuple, Any


class Module(nn.Module, metaclass=abc.ABCMeta):
    """
    Base kernel class
    """

    def __init__(self):
        super().__init__()

    @property
    @abc.abstractmethod
    def prms(self) -> Any:
        pass


class NoneClass(Module):

    def __init__(self):
        super().__init__()

    @property
    def prms(self) -> Any:
        return

    @property
    def msg(self):
        return ''
