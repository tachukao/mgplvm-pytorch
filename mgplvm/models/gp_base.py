import torch
from ..base import Module
from torch import Tensor
import abc
from typing import Tuple, List, Optional, Union


class GPBase(Module, metaclass=abc.ABCMeta):
    """Base GP model class."""

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def elbo(self, y, x, sample_idxs, m):
        return
