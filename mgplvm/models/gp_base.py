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
    def elbo(self,
             y: Tensor,
             x: Tensor,
             sample_idxs: Optional[List[int]] = None,
             m: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        return
