import abc
import math
import torch
from torch import nn, Tensor
from ..utils import softplus, inv_softplus
from .kernel import Kernel
from typing import Tuple, List
import numpy as np


class Stationary(Kernel, metaclass=abc.ABCMeta):

    def __init__(self,
                 n: int,
                 distance,
                 d=None,
                 ell=None,
                 scale=None,
                 learn_scale=True,
                 Y: np.ndarray = None,
                 eps: float = 1e-6,
                 ell_byneuron: bool = True):
        """
        Parameters
        ----------
        n : int 
            number of batches (neurons)
        distance : 
            distance function
        d : Optional[int]
            dimension of the input variables
            if provided, there is a separate length scale for each input dimension
        ell: Optional[np.ndarray]
            lengthscale hyperparameter
            it should have dimensions n x d if d is not None and n if d is None
        scale : Optional[np.ndarray]
            scale hyperparameter (std)
            it should have dimension n 
        learn_scale : bool
            optimises the scale hyperparameter if true
        Y : Optional[np.ndarray]
            data matrix used for initializing the scale hyperparameter
        eps: float
            minimum ell
        """

        super(Stationary, self).__init__()

        self.eps = eps

        if scale is not None:
            _scale_sqr = torch.tensor(scale,
                                      dtype=torch.get_default_dtype()).square()
        elif Y is not None:
            _scale_sqr = torch.tensor(1 * np.mean(Y**2, axis=(0, -1)))
        else:
            _scale_sqr = torch.ones(n,)

        self._scale_sqr = nn.Parameter(data=inv_softplus(_scale_sqr),
                                       requires_grad=learn_scale)

        self.ard = (d is not None)
        if ell is None:
            if d is None:
                _ell = inv_softplus(2 * torch.ones(n,))
            elif ell_byneuron:
                assert (d is not None)
                _ell = inv_softplus(2 * torch.ones(n, d))
            else:
                _ell = inv_softplus(2 * torch.ones(1, d))
        else:
            if d is not None:
                assert ell.shape[-1] == d
            _ell = inv_softplus(
                torch.tensor(ell, dtype=torch.get_default_dtype()))

        self._ell = nn.Parameter(data=_ell - self.eps, requires_grad=True)

        self.distance = distance

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
        vector (scale, scale, ..., scale)
        """
        shp = list(x.shape)
        del shp[-2]
        scale_sqr = self.scale_sqr[:, None]
        return torch.ones(shp).to(scale_sqr.device) * scale_sqr

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
        For a stationary quad exp kernel, the trace is scale * mx
        """
        scale_sqr = self.scale_sqr
        return torch.ones(x.shape[:-2]).to(
            scale_sqr.device) * scale_sqr * x.shape[-1]

    @property
    def prms(self) -> Tuple[Tensor, Tensor]:
        return self.scale_sqr, self.ell

    @property
    def scale_sqr(self) -> Tensor:
        return softplus(self._scale_sqr)

    @property
    def scale(self) -> Tensor:
        return (self.scale_sqr + 1e-20).sqrt()

    @property
    def ell(self) -> Tensor:
        return softplus(self._ell) + self.eps

    @property
    def msg(self):
        return (' scale {:.3f} | ell {:.3f} |').format(self.scale.mean().item(),
                                                       self.ell.mean().item())


class QuadExp(Stationary):

    def __init__(self,
                 n: int,
                 distance,
                 d=None,
                 ell=None,
                 scale=None,
                 learn_scale=True,
                 Y: np.ndarray = None,
                 eps: float = 1e-6,
                 ell_byneuron: bool = True):
        """
        Quadratic exponential kernel

        Parameters
        ----------
        n : int 
            number of batches (neurons)
        distance : 
            distance function
        d : Optional[int]
            dimension of the input variables
            if provided, there is a separate length scale for each input dimension
        ell: Optional[np.ndarray]
            lengthscale hyperparameter
            it should have dimensions n x d if d is not None and n if d is None
        scale : Optional[np.ndarray]
            scale hyperparameter
            it should have dimension n 
        learn_scale : bool
            optimises the scale hyperparameter if true
        Y : Optional[np.ndarray]
            data matrix used for initializing the scale hyperparameter
        eps : float
            minimum ell
        """

        super(QuadExp, self).__init__(n, distance, d, ell, scale, learn_scale,
                                      Y, eps, ell_byneuron)

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
        scale_sqr, ell = self.prms
        if self.ard:
            ell = ell[:, :, None]  #(n x d x 1)
        else:
            ell = ell[:, None, None]  #(n x 1 x 1)
        distance = self.distance(x, y, ell=ell)  # dims (... n x mx x my)
        kxy = scale_sqr[:, None, None] * torch.exp(-0.5 * distance)
        return kxy


class Exp(QuadExp):

    def __init__(self,
                 n: int,
                 distance,
                 d=None,
                 ell=None,
                 scale=None,
                 learn_scale=True,
                 Y: np.ndarray = None,
                 eps: float = 1E-6):
        super().__init__(n, distance, d, ell, scale, learn_scale, Y=Y, eps=eps)

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
            exponential kernel with dims (... n x mx x my)

        """
        scale_sqr, ell = self.prms
        if self.ard:
            ell = ell[:, :, None]  #(n x d x 1) / (1 x d x 1)
        else:
            ell = ell[:, None, None]  #(n x 1 x 1)
        distance = self.distance(x, y, ell=ell)  # dims (... n x mx x my)

        # NOTE: distance means squared distance ||x-y||^2 ?
        stable_distance = torch.sqrt(distance + 1e-20)  # numerically stabilized
        kxy = scale_sqr[:, None, None] * torch.exp(-stable_distance)
        return kxy


class Matern(Stationary):

    def __init__(self,
                 n: int,
                 distance,
                 d=None,
                 nu=1.5,
                 ell=None,
                 scale=None,
                 learn_scale=True,
                 Y=None,
                 eps: float = 1E-6):
        '''
        Parameters
        ----------
        n : int 
            number of neurons/readouts
        distance :
            a squared distance function

        Note
        -----
        based on the gpytorch implementation:
        https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/kernels/matern_kernel.py
        '''
        super().__init__(n, distance, d, ell, scale, learn_scale, Y=Y, eps=eps)

        if nu not in (0.5, 1.5, 2.5):
            raise Exception("only nu=0.5, 1.5, 2.5 implemented")

        self.nu = nu

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
            matern kernel with dims (... n x mx x my)

        """

        scale_sqr, ell = self.prms
        if self.ard:
            ell = ell[:, :, None]
        else:
            ell = ell[:, None, None]
        distance = (self.distance(x, y, ell=ell) + 1E-20).sqrt()

        # NOTE: distance means squared distance ||x-y||^2 ?
        z1 = torch.exp(-math.sqrt(self.nu * 2) * distance)
        if self.nu == 0.5:
            z2 = 1
        elif self.nu == 1.5:
            z2 = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            z2 = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
        return scale_sqr[:, None, None] * z1 * z2

    @property
    def msg(self):
        return (' nu {:.1f} | scale {:.3f} | ell {:.3f} |').format(
            self.nu,
            self.scale.mean().item(),
            self.ell.mean().item())
