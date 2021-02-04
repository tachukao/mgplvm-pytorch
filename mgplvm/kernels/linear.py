import torch
from torch import nn, Tensor
from .kernel import Kernel
from typing import Tuple, List
import numpy as np


class Linear(Kernel):
    name = "Linear"

    def __init__(self,
                 n: int,
                 d: int,
                 alpha=None,
                 learn_weights=False,
                 learn_alpha=False,
                 Y=None):
        '''
        n is number of neurons/readouts
        distance is the distance function used
        d is the dimensionality of the group parameterization
        scaling determines wheter an output scale parameter is learned for each neuron
        
        learn_weights: learn PCA/FA style weights
        learn_alpha: learn an output scaling parameter (similar to the RBF signal variance)
        '''
        super().__init__()

        if alpha is not None:
            _alpha = torch.tensor(alpha)
        elif Y is not None:  # <Y^2> = alpha^2 * d * <x^2> + <eps^2> = alpha^2 * d + sig_noise^2
            _alpha = torch.tensor(np.sqrt(np.var(
                Y, axis=(0, 2)) / d)) * 0.5  #assume half signal half noise
        else:
            _alpha = torch.ones(n,)  #one per neuron
        self._alpha = nn.Parameter(data=_alpha, requires_grad=learn_alpha)

        self.learn_weights = learn_weights
        #W = torch.ones(n, d ) #full weight matrix
        W = torch.randn(n, d) * 0.1
        self.W = nn.Parameter(data=W, requires_grad=learn_weights)

    def diagK(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            input tensor of dims (... n x d x mx)

        Returns
        -------
        diagK : Tensor
            diagonal of kernel K(x,x) with dims (... n x mx )

        Note
        ----
        For a linear kernel, the diagonal is a mx-dimensional 
        vector (||x_1||^2, ||x_2||^2, ..., ||x_mx||^2)
        """
        W, alpha = self.prms

        if self.learn_weights:
            x = (W[:, :, None] * x).sum(dim=-2, keepdim=True)  # n x 1 x mx

        sqr_alpha = torch.square(alpha)[:, None, None].to(x.device)
        diag = (sqr_alpha * torch.square(x)).sum(dim=-2)

        return diag

    def trK(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            input tensor of dims (... n x d x mx)

        Returns
        -------
        trK : Tensor
            trace of kernel K(x,x) with dims (... n)

        Note
        ----
        For a stationary quad exp kernel, the trace is alpha^2 * mx
        """
        return self.diagK(x).sum(dim=-1)

    def K(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            input tensor of dims (... n x d x mx)
        y : Tensor
            input tensor of dims (... n x d x my)

        Returns
        -------
        kxy : Tensor
            linear kernel with dims (... n x mx x my)
        
        
        W: nxd
        X: n x d x mx
        
        x: d x mx
        x^T w_n w_n^T y (mx x my)
        
        
        K_n(x, y) = w_n X^T (mx x my)
        K(X, Y) (n x mx x my)

        """
        W, alpha = self.prms

        if self.learn_weights:
            x = (W[:, :, None] * x).sum(dim=-2, keepdim=True)  # n x 1 x mx
            y = (W[:, :, None] * y).sum(dim=-2, keepdim=True)  # n x 1 x my

        sqr_alpha = torch.square(alpha)[:, None, None].to(x.device)
        distance = x.transpose(-1, -2).matmul(y)

        kxy = sqr_alpha * distance
        return kxy

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.K(x, y)

    @property
    def prms(self) -> Tuple[Tensor, Tensor]:
        return self.W, self._alpha

    @property
    def alpha(self) -> Tensor:
        return torch.abs(self._alpha)

    @property
    def msg(self):
        W, alpha = self.prms
        return (' W {:.3f} | alpha {:.3f} |').format(
            (W**2).mean().sqrt().item(), (alpha**2).mean().sqrt().item())


#class Combination(Kernel):
#
#    def __init__(self, kernels: List[Kernel]):
#        """
#        Combination Kernels
#
#        Parameters
#        ----------
#        kernels : list of kernels
#
#        Notes
#        -----
#        Implementation largely follows thats described in
#        https://github.com/GPflow/GPflow/blob/develop/gpflow/kernels/base.py
#        """
#        super().__init__()
#        self.kernels = kernels
#
#    def forward(self, x: List[Tensor], y: List[Tensor]) -> Tensor:
#        return self._reduce([k(x, y) for (k, x, y) in zip(self.kernels, x, y)])
#
#    @abc.abstractmethod
#    def _reduce(self, x: List[Tensor]) -> Tensor:
#        pass
#
#    @property
#    def prms(self) -> List[Tuple[Tensor]]:
#        return [k.prms for k in self.kernels]
#
#
#class Sum(Combination):
#
#    def _reduce(self, x: List[Tensor]) -> Tensor:
#        return torch.sum(torch.stack(x, dim=0), dim=0)
#
#    def trK(self, x: Tensor) -> Tensor:
#        """
#        sum_i(alpha_1^2 + alpha_2^2)
#        """
#        alphas = [k.prms[0] for k in self.kernels]
#        sqr_alphas = [torch.square(alpha) for alpha in alphas]
#        sqr_alpha = torch.stack(sqr_alphas).sum(dim=0)
#        return torch.ones(x[0].shape[:-2]).to(
#            sqr_alpha.device) * sqr_alpha * x[0].shape[-1]
#
#
#class Product(Combination):
#
#    def _reduce(self, x: List[Tensor]):
#        return torch.prod(torch.stack(x, dim=0), dim=0)
#
#    def trK(self, x: Tensor) -> Tensor:
#        """
#        sum_i(alpha_1^2 * alpha_2^2)
#        """
#        alphas = [k.prms[0] for k in self.kernels]
#        sqr_alphas = [torch.square(alpha) for alpha in alphas]
#        sqr_alpha = torch.stack(sqr_alphas).prod(dim=0)
#        return torch.ones(x[0].shape[:-2]).to(
#            sqr_alpha.device) * sqr_alpha * x[0].shape[-1]
