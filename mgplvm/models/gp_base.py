import torch
from ..base import Module
from torch import Tensor
import abc
from typing import Tuple, List, Optional, Union


class GpBase(Module, metaclass=abc.ABCMeta):
    """Base p(Y|X) class"""

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def elbo(self, y, x, sample_idxs, m):
        return
