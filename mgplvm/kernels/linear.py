import torch
from torch import nn, Tensor
from .kernel import Kernel
from typing import Tuple, List
import numpy as np


class Linear(Kernel):
    name = "Linear"

    def __init__(self, n: int, d: int, scale=None, learn_scale=False, Y=None):
        '''
        n is number of neurons/readouts
        distance is the distance function used
        d is the dimensionality of the group parameterization
        scaling determines wheter an output scale parameter is learned for each neuron
        
        learn_scale : learn an output scaling parameter (similar to the RBF signal variance)

        """
        Note
        ----
        W: nxd
        X: n x d x mx
        
        x: d x mx
        x^T w_n w_n^T y (mx x my)
        
        K_n(x, y) = w_n X^T (mx x my)
        K(X, Y) (n x mx x my)
        '''
        super().__init__()

        if scale is not None:
            _scale = torch.tensor(scale)
        elif Y is not None:  # <Y^2> = scale * d * <x^2> + <eps^2> = scale * d + sig_noise^2
            _scale = torch.tensor(np.sqrt(np.var(
                Y, axis=(0, 2)) / d)) * 0.5  #assume half signal half noise
        else:
            _scale = torch.ones(n,)  #one per neuron
        self._scale = nn.Parameter(data=_scale, requires_grad=learn_scale)

    def diagK(self, x: Tensor) -> Tensor:
        diag = (self.scale_sqr[:, None, None] * (x**2)).sum(dim=-2)
        return diag

    def trK(self, x: Tensor) -> Tensor:
        return self.diagK(x).sum(dim=-1)

    def K(self, x: Tensor, y: Tensor) -> Tensor:
        distance = x.transpose(-1, -2).matmul(y)
        kxy = self.scale_sqr[:, None, None] * distance
        return kxy

    @property
    def prms(self) -> Tensor:
        return self.scale

    @property
    def scale_sqr(self) -> Tensor:
        return self._scale.square()

    @property
    def scale(self) -> Tensor:
        return self._scale.abs()

    @property
    def msg(self):
        return ('scale {:.3f} |').format(self.scale.mean().item())
