import torch
from torch import nn, Tensor
from .kernel import Kernel
from typing import Tuple, List
import numpy as np
from ..utils import softplus, inv_softplus


class Linear(Kernel):
    name = "Linear"

    def __init__(self,
                 n: int,
                 d: int,
                 scale=None,
                 learn_scale=True,
                 Y=None,
                 ard=False,
                 Poisson=False):
        '''
        n is number of neurons/readouts
        d is the dimensionality of the group parameterization
        scale is an output scale parameter for each neuron
        
        learn_scale : learn an output scaling parameter (similar to the RBF signal variance)

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
            _scale_sqr = torch.tensor(scale).square()
        elif (
                Y is not None
        ) and learn_scale:  # <Y^2> = scale * d * <x^2> + <eps^2> = scale * d + sig_noise^2
            _scale_sqr = torch.tensor(np.var(Y, axis=(0, 2)) / d) * (
                0.5**2)  #assume half signal half noise
            if Poisson:
                _scale_sqrt /= 100
        else:
            _scale_sqr = torch.ones(n,)  #one per neuron
        self._scale_sqr = nn.Parameter(data=inv_softplus(_scale_sqr),
                                       requires_grad=learn_scale)

        if ard and (not learn_scale) and (Y is not None):
            _input_scale = torch.ones(d) * (np.std(Y) / np.sqrt(d) * 0.5)
            if Poisson:
                _input_scale /= 100
        else:
            _input_scale = torch.ones(d)

        _input_scale = inv_softplus(_input_scale)
        self._input_scale = nn.Parameter(data=_input_scale, requires_grad=ard)

    def diagK(self, x: Tensor) -> Tensor:
        diag = (self.scale_sqr[:, None, None] *
                (self.reweight(x)**2)).sum(dim=-2)
        return diag

    def trK(self, x: Tensor) -> Tensor:
        return self.diagK(self.reweight(x)).sum(dim=-1)

    def K(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            input tensor of dims (... n_samples x n x d x mx)
        y : Tensor
            input tensor of dims (... n_samples x n x d x mx)

        Returns
        -------
        trK : Tensor
            trace of kernel K(x,x) with dims (... n)
        """

        # compute x dot y with latent reweighting
        dot = self.reweight(x).transpose(-1, -2).matmul(self.reweight(y))
        # multiply by scale factor
        kxy = self.scale_sqr[:, None, None] * dot
        return kxy

    def reweight(self, x: Tensor) -> Tensor:
        """re-weight the latent dimensions"""
        x = self.input_scale[:, None] * x
        return x

    @property
    def prms(self) -> Tuple[Tensor, Tensor]:
        return self.scale_sqr, self.input_scale

    @property
    def scale_sqr(self) -> Tensor:
        return softplus(self._scale_sqr)

    @property
    def scale(self) -> Tensor:
        return (self.scale_sqr + 1e-20).sqrt()

    @property
    def input_scale(self) -> Tensor:
        return softplus(self._input_scale)

    @property
    def msg(self):
        return ('scale {:.3f} | input_scale {:.3f} |').format(
            self.scale.mean().item(),
            self.input_scale.mean().item())
