import torch
from torch import nn, Tensor
from ..utils import softplus, inv_softplus
from .kernel import Kernel
from typing import Tuple, List
import numpy as np


class QuadExp(Kernel):
    name = "QuadExp"

    def __init__(self,
                 n: int,
                 distance,
                 d=None,
                 ell=None,
                 scale=None,
                 learn_scale=True,
                 Y: np.ndarray = None):
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
        """

        super(QuadExp, self).__init__()

        if scale is not None:
            _scale = torch.tensor(scale, dtype=torch.get_default_dtype()).sqrt()
        elif Y is not None:
            _scale = torch.tensor(np.mean(Y**2, axis=(0, -1))).sqrt()
        else:
            _scale = torch.ones(n,)

        self._scale = nn.Parameter(data=_scale, requires_grad=learn_scale)

        self.ard = d is not None
        if ell is None:
            if d is None:
                _ell = inv_softplus(torch.ones(n,))
            else:
                assert (d is not None)
                _ell = inv_softplus(torch.ones(n, d))
        else:
            if d is not None:
                assert ell.shape[-1] == d
            ell = inv_softplus(
                torch.tensor(_ell, dtype=torch.get_default_dtype()))

        self._ell = nn.Parameter(data=_ell, requires_grad=True)

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
        scale, ell = self.prms
        if self.ard:
            ell = ell[:, None, :]
        else:
            ell = ell[:, None, None]
        distance = self.distance(x / ell, y / ell)  # dims (... n x mx x my)
        scale = scale[:, None, None]
        kxy = scale * torch.exp(-0.5 * distance)
        return kxy

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
        scale = self.scale[:, None]
        shp = list(x.shape)
        del shp[-2]
        return torch.ones(shp).to(scale.device) * scale

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
        scale = self.scale
        return torch.ones(x.shape[:-2]).to(scale.device) * scale * x.shape[-1]

    @property
    def prms(self) -> Tuple[Tensor, Tensor]:
        return self.scale, self.ell

    @property
    def scale(self) -> Tensor:
        return self._scale.square()

    @property
    def ell(self) -> Tensor:
        return softplus(self._ell)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.K(x, y)

    @property
    def msg(self):
        alpha_mag, ell_mag = [val.mean().item() for val in self.prms]
        return (' alpha_sqr {:.3f} | ell {:.3f} |').format(
            alpha_mag**2, ell_mag)


class Exp(QuadExp):
    name = "Exp"

    def __init__(self,
                 n: int,
                 distance,
                 d=None,
                 ell=None,
                 scale=None,
                 learn_scale=True,
                 Y: np.ndarray = None):
        super().__init__(n, distance, d, ell, scale, learn_scale, Y=Y)
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
            exponential kernel with dims (... n x mx x my)

        """
        scale, ell = self.prms
        expand_scale = scale[:, None, None]
        if self.ard:
            expand_ell = ell[:, None, :]
        else:
            expand_ell = ell[:, None, None]
        distance = self.distance(x / expand_ell,
                                 y / expand_ell)  # dims (... n x mx x my)

        # NOTE: distance means squared distance ||x-y||^2 ?
        stable_distance = torch.sqrt(distance + 1e-12)  # numerically stabilized
        kxy = expand_scale * torch.exp(-stable_distance)
        return kxy

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.K(x, y)

    @property
    def msg(self):
        return (' scale {:.3f} | ell {:.3f} |').format(self.scale.mean().item(),
                                                       self.ell.mean().item())


class Matern(QuadExp):
    name = "Matern"

    def __init__(self,
                 n: int,
                 distance,
                 d=None,
                 nu=3 / 2,
                 ell=None,
                 scale=None,
                 learn_scale=True):
        '''
        n is number of neurons/readouts
        distance is a squared distance function
        '''
        super().__init__(n, distance, d, ell, scale, learn_scale)

        assert nu in [3 / 2, 5 / 2], "only nu=3/2 and nu=5/2 implemented"
        if nu == 3 / 2:
            self.k_r = self.k_r_3_2
        elif nu == 5 / 2:
            self.k_r = self.k_r_5_2

        self.nu = nu
        self.distance_sqr = distance

    def distance(self, x, y):
        d_sqr = self.distance_sqr(x, y)
        print(torch.min(d_sqr), torch.max(d_sqr))
        return torch.sqrt(d_sqr)

    @staticmethod
    def k_r_3_2(scale, r, ell):
        sqrt3_r_l = np.sqrt(3) * r / ell
        kxy = scale * (1 + sqrt3_r_l) * torch.exp(-sqrt3_r_l)
        return kxy

    @staticmethod
    def k_r_5_2(scale, r, ell):
        sqrt5_r_l = np.sqrt(5) * r / ell
        sqr_term = 5 / 3 * torch.square(r) / torch.square(ell)
        kxy = scale * (1 + sqrt5_r_l + sqr_term) * torch.exp(-sqrt5_r_l)
        return kxy

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

        scale, ell = self.prms
        scale = scale[:, None, None]
        ell = ell[:, None, None]
        r = self.distance(x, y)  # dims (... n x mx x my)
        print(torch.min(r), torch.max(r))

        return self.k_r(scale, r, ell)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.K(x, y)

    @property
    def msg(self):
        return (' nu {:.1f} | scale {:.3f} | ell {:.3f} |').format(
            self.nu,
            self.scale.mean().item(),
            self.ell.mean().item())
