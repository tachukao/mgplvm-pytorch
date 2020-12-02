import abc
import torch
from torch import nn, Tensor
from .utils import softplus, inv_softplus
from .base import Module
from typing import Tuple, List, Sequence
import numpy as np


class Kernel(Module, metaclass=abc.ABCMeta):
    """
    Base kernel class
    """
    def __init__(self):
        super().__init__()


class Combination(Kernel):
    def __init__(self, kernels: List[Kernel]):
        """
        Combination Kernels

        Parameters
        ----------
        kernels : list of kernels

        Notes
        -----
        Implementation largely follows thats described in 
        https://github.com/GPflow/GPflow/blob/develop/gpflow/kernels/base.py
        """
        super().__init__()
        self.kernels = kernels

    def forward(self, x: List[Tensor], y: List[Tensor]) -> Tensor:
        return self._reduce([k(x, y) for (k, x, y) in zip(self.kernels, x, y)])

    @abc.abstractmethod
    def _reduce(self, x: List[Tensor]) -> Tensor:
        pass

    @property
    def prms(self) -> List[Tuple[Tensor]]:
        return [k.prms for k in self.kernels]


class Sum(Combination):
    def _reduce(self, x: List[Tensor]) -> Tensor:
        return torch.sum(torch.stack(x, dim=0), dim=0)

    def trK(self, x: Tensor) -> Tensor:
        """
        sum_i(alpha_1^2 + alpha_2^2)
        """
        alphas = [k.prms[0] for k in self.kernels]
        sqr_alphas = [torch.square(alpha) for alpha in alphas]
        sqr_alpha = torch.stack(sqr_alphas).sum(dim=0)
        return torch.ones(x[0].shape[:-2]).to(
            sqr_alpha.device) * sqr_alpha * x[0].shape[-1]


class Product(Combination):
    def _reduce(self, x: List[Tensor]):
        return torch.prod(torch.stack(x, dim=0), dim=0)

    def trK(self, x: Tensor) -> Tensor:
        """
        sum_i(alpha_1^2 * alpha_2^2)
        """
        alphas = [k.prms[0] for k in self.kernels]
        sqr_alphas = [torch.square(alpha) for alpha in alphas]
        sqr_alpha = torch.stack(sqr_alphas).prod(dim=0)
        return torch.ones(x[0].shape[:-2]).to(
            sqr_alpha.device) * sqr_alpha * x[0].shape[-1]


class QuadExpBase(Kernel):
    def __init__(self, n: int, ell=None, alpha=None):
        super().__init__()

        alpha = inv_softplus(torch.ones(
            n, )) if alpha is None else inv_softplus(
                torch.tensor(alpha, dtype=torch.get_default_dtype()))
        self.alpha = nn.Parameter(data=alpha, requires_grad=True)

        ell = inv_softplus(torch.ones(n, )) if ell is None else inv_softplus(
            torch.tensor(ell, dtype=torch.get_default_dtype()))
        self.ell = nn.Parameter(data=ell, requires_grad=True)

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
        For a stationary quad exp kernel, the diagonal is a mx-dimensional 
        vector (alpha^2, alpha^2, ..., alpha^2)
        """
        alpha, _ = self.prms
        sqr_alpha = torch.square(alpha)[:, None]
        shp = list(x.shape)
        del shp[-2]
        return torch.ones(shp).to(sqr_alpha.device) * sqr_alpha

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
        alpha, _ = self.prms
        sqr_alpha = torch.square(alpha)
        return torch.ones(x.shape[:-2]).to(
            sqr_alpha.device) * sqr_alpha * x.shape[-1]

    @abc.abstractmethod
    def K(self, x: Tensor, y: Tensor) -> Tensor:
        pass

    @property
    def prms(self) -> Tuple[Tensor, Tensor]:
        alpha = softplus(self.alpha)
        ell = softplus(self.ell)
        return alpha, ell


class QuadExp(QuadExpBase):
    name = "QuadExp"

    def __init__(self, n: int, distance, ell=None, alpha=None):
        super().__init__(n, ell, alpha)
        self.distance = distance

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
            quadratic exponential kernel with dims (... n x mx x my)

        """
        alpha, ell = self.prms
        distance = self.distance(x, y)  # dims (... n x mx x my)
        sqr_alpha = torch.square(alpha)[:, None, None]
        sqr_ell = torch.square(ell)[:, None, None]
        kxy = sqr_alpha * torch.exp(-0.5 * distance / sqr_ell)
        return kxy

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.K(x, y)

    @property
    def msg(self):
        alpha_mag, ell_mag = [
            np.mean(val.data.cpu().numpy()) for val in self.prms
        ]
        return (' alpha_sqr {:.3f} | ell {:.3f} |').format(
            alpha_mag**2, ell_mag)


class QuadExpARD(QuadExpBase):
    name = "QuadExpARD"

    def __init__(self, n: int, d: int, ard_distance):
        super().__init__(n)
        self.ell = nn.Parameter(data=softplus(1 * torch.randn(n, d)),
                                requires_grad=True)
        self.ard_distance = ard_distance

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
            quadratic exponential ARD kernel with dims (... n x mx x my)

        """
        alpha, ell = self.prms
        ard_distance = self.ard_distance(x, y)
        sqr_alpha = torch.square(alpha)[:, None, None]
        sqr_ell = torch.square(ell)[..., None, None]
        kxy = sqr_alpha * torch.exp(-0.5 * (ard_distance / sqr_ell).sum(-3))
        return kxy

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.K(x, y)

    @property
    def msg(self):
        alpha_mag, ell_mag = [
            np.mean(val.data.cpu().numpy()) for val in self.prms
        ]
        return (' alpha_sqr {:.3f} | ell {:.3f} |').format(
            alpha_mag**2, ell_mag)


class Linear(Kernel):
    name = "Linear"

    def __init__(self, n: int, distance, scaling=False):
        '''
        n is number of neurons/readouts
        distance is the distance function used
        scaling determines wheter an output scale parameter is learned for each neuron
        '''
        super().__init__()

        self.distance = distance

        scale = inv_softplus(torch.ones(n, ))
        if scaling:
            self.scale = nn.Parameter(data=scale, requires_grad=True)
        else:
            self.scale = scale

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
        scale = self.prms
        sqr_scale = torch.square(scale)[:, None, None].to(x.device)
        diag = (sqr_scale * torch.square(x)).sum(axis=-2)

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
        return self.diagK(x).sum(axis=-1)

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
            quadratic exponential kernel with dims (... n x mx x my)

        """
        scale = self.prms
        sqr_scale = torch.square(scale)[:, None, None].to(x.device)
        distance = self.distance(x, y)  # dims (... n x mx x my)

        kxy = sqr_scale * distance
        return kxy

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.K(x, y)

    @property
    def prms(self) -> Tensor:
        return softplus(self.scale)

    @property
    def msg(self):
        scale = self.prms
        return (' scale {:.3f} |').format(scale.mean())
