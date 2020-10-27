import abc
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from mgplvm.utils import softplus, inv_softplus
from .base import Module
from .kernels import Kernel, Combination
from .inducing_variables import InducingPoints
from typing import Tuple, List, Optional, Union

jitter: float = 1E-8
log2pi: float = np.log(2 * np.pi)


class SgpBase(Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 kernel: Kernel,
                 n: int,
                 m: int,
                 n_inducing: int,
                 sigma: Optional[Tensor] = None):
        """
        __init__ method for Base Sparse GP Class

        Parameters
        ----------
        n : int
            number of neurons
        m : int
            number of conditions
        n_inducing : int
            number of inducing points
        sigma : optional Tensor
            initialization of the output noise standard deviation
            i.e., noise variance is sigma^2
        """
        super().__init__()
        self.n = n
        self.m = m
        self.n_inducing = n_inducing
        sigma = torch.ones(n,) * 0.2 if sigma is None else torch.tensor(
            sigma, dtype=torch.get_default_dtype())
        self.sigma = nn.Parameter(data=inv_softplus(sigma), requires_grad=True)
        self.kernel = kernel

    @abc.abstractmethod
    def _expand(self, z, x):
        pass

    def elbo(self,
             n_samples: int,
             n_b: int,
             y: Tensor,
             x: Tensor,
             tosum=True) -> Tensor:
        """
        Parameters
        ----------
        n_samples : int
            number of data samples
        n_b : int
            batch size
        y : Tensor
            data tensor with dimensions (n x m x n_samples)
        x : Tensor (single kernel) or Tensor list (product kernels)
            input tensor(s) with dimensions (n_b x d x m)

        Returns
        -------
        evidence lower bound : torch.Tensor

        Notes
        -----
        Implementation largely follows derivation of the ELBO presented in 
        https://gpflow.readthedocs.io/en/develop/notebooks/theory/SGPR_notes.html

        """

        sigma, z = self.prms  # noise cov and inducing points
        kernel = self.kernel
        m = self.m  # conditions
        n_inducing = self.n_inducing  # inducing points
        sqr_sigma = sigma * sigma  # noise variance

        # expand z and x so they have the right dimensionality for the kernels
        # after expansion, u and x are four-dimensional tensors with dimensions
        # that correspond to (batch, neuron, manif dim, condition)
        # z and x are both tensors (single kernel) or lists of tensors (product kernels)
        # z tensors have dimensions (1 x n x d x n_inducing)
        # x tensors have dimensions (n_b x n x d x m)
        z, x = self._expand(z, x)

        kzz = kernel(z, z)  # dims: (1 x n x n_inducing x n_inducing)
        kzx = kernel(z, x)  # dims: (n_b x n x n_inducing x m)
        trkxx = kernel.trK(x)  # dims: (n_b x n)
        e = torch.eye(n_inducing).to(kzz.device)

        # [ l ] dims: (1 x n x n_inducing x n_inducing)
        l = torch.cholesky(kzz + (e * jitter), upper=False)
        # [ a ] dims: (n_b x n x n_inducing x m)
        a = torch.triangular_solve(kzx, l, upper=False)[0] / sigma
        # [ aat ] dims: (n_b x n x n_inducing x n_inducing)
        aat = torch.matmul(a, a.permute(0, 1, 3, 2))
        # [ b ] dims: (n_b x n x n_inducing x n_inducing)
        b = aat + e
        # [ lb ] dims: (n_b x n x n_inducing x n_inducing)
        lb = torch.cholesky(b, upper=False)
        # [ y ] dims (n x m x n_samples), [ aerr ] dims (n_b x n x n_inducing x n_samples)
        aerr = torch.matmul(a, y)
        # [ c ] dims (n_b x n x n_inducing x n_samples)
        c = torch.triangular_solve(aerr, lb, upper=False)[0] / sigma

        # [ l0 ] dims: ()
        l0 = -0.5 * m * log2pi
        # [ l1 ] dims: (n_b x n x n_inducing)
        l1 = -torch.log(torch.diagonal(lb, dim1=-2, dim2=-1))
        # [ l2 ] dims: (n, 1, 1)
        l2 = -m * torch.log(sigma)
        # [ l3 ] dims: (n x m x n_samples)
        l3 = -0.5 * torch.square(y / sigma)
        # [ l4 ] dims: (n_b x n x n_inducing x n_samples)
        l4 = 0.5 * torch.square(c)
        # [ l5 ] dims: (n_b x n )
        l5 = -0.5 * trkxx / sqr_sigma[:, 0, 0]
        # [ l6 ] dims: (n_b x n x n_inducing )
        l6 = 0.5 * torch.diagonal(aat, dim1=-2, dim2=-1)

        if tosum:  # sum over batches
            # scale l0, l1, ..., l6 so that each represent a total of
            # (n x n_b x n_samples) "samples"
            l0 = l0 * n_b * self.n * n_samples
            l1 = l1.sum() * n_samples
            l2 = l2.sum() * n_b * n_samples
            l3 = l3.sum() * n_b
            l4 = l4.sum()
            l5 = l5.sum() * n_samples
            l6 = l6.sum() * n_samples

            return torch.sum(l0 + l1 + l2 + l5 + l6 + l3 + l4)

        else:  # don't sum over batches
            l0 = torch.tensor(l0 * self.n * n_samples).to(l1.device)  # (1,)
            l1 = l1.sum(dim=[1, 2]) * n_samples  # (n_b,)
            l2 = (l2.sum() * n_samples).reshape(1)  # (1,)
            l3 = l3.sum().reshape(1)  # (1,)
            l4 = l4.sum(dim=[1, 2, 3])  # (n_b,)
            l5 = l5.sum(dim=1) * n_samples  # (n_b,)
            l6 = l6.sum(dim=[1, 2]) * n_samples  # (n_b,)

            return l0 + l1 + l2 + l3 + l4 + l5 + l6

    def prediction(self, y: Tensor, x: Tensor, s: Tensor) -> Tensor:
        """
        Parameters
        ----------
        y : Tensor
            data tensor with dimensions (n x m x n_samples)
        x : Tensor (single kernel) or Tensor list (product kernels)
            input tensor(s) with dimensions (n_b x d x m)
        s : Tensor (single kernel) or Tensor list (product kernels)
            test input tensor(s) with dimensions (1 x d x npred)

        Returns
        -------
        mu : Tensor 
            mean of predictive density at test inputs [ s ]
        v : Tensor 
            covariance of predictive density at test inputs [ s ]

        Notes
        -----
        Implementation largely follows derivation of the predictive density presented in 
        https://gpflow.readthedocs.io/en/develop/notebooks/theory/SGPR_notes.html

        """

        sigma, z = self.prms
        kernel = self.kernel

        # see ELBO for explanation of _expand
        z, x = self._expand(z, x)
        kzz = kernel(z, z)  # dims: (1 x n x n_inducing x n_inducing)
        kzx = kernel(z, x)  # dims: (1 x n x n_inducing x m)
        kzs = kernel(z, s)  # dims: (1 x n x n_inducing x npred)
        kss = kernel(s, s)  # dims: ((1) x n x n_s x n_s)
        e = torch.eye(self.n_inducing).to(kzz.device)

        #alpha, _ = kernel.prms
        #alpha_sqr = torch.square(alpha)
        # [ l ] has dims: (1 x n x n_inducing x n_inducing)
        l = torch.cholesky(kzz + (jitter * e), upper=False)
        # [ a ] has dims: (n_b x n x n_inducing x m)
        a = torch.triangular_solve(kzx, l, upper=False)[0] / sigma
        # [ aat ] has dims: (n_b x n x n_inducing x n_inducing)
        aat = torch.matmul(a, a.permute(0, 1, 3, 2))
        # [ b ] has dims: (n_b x n x n_inducing x n_inducing)
        b = aat + e
        # [ lb ] has dims: (n_b x n x n_inducing x n_inducing)
        lb = torch.cholesky(b, upper=False)
        # [ z ] has dims: (n_b x n x n_inducing x m)
        z = torch.matmul(a, y) / sigma
        # [ c ] has dims: (n_b x n x n_inducing x n_samples)
        c = torch.triangular_solve(z, lb, upper=False)[0]

        # [ alpha ] has dims: (1 x n x n_inducing x n_s)
        alpha = torch.triangular_solve(kzs, l, upper=False)[0]
        # [ alphat ] has dims: (1 x n x n_s x n_inducing)
        alphat = alpha.permute(0, 1, 3, 2)
        # [ beta ] has dims: (n_b x n x n_inducing x n_s)
        beta = torch.triangular_solve(alpha, lb, upper=False)[0]
        # [ betat ] has dims: (n_b x n x n_s x n_inducing)
        betat = beta.permute(0, 1, 3, 2)
        # [ mu ] has dims: (n_b x n x n_s x n_samples)
        mu = torch.matmul(betat, c)
        # [ v0 ] has dims: (1 x n x n_s x n_s)
        v0 = kss
        # [ v1 ] has dims: (1 x n x n_inducing x n_inducing)
        v1 = torch.matmul(alphat, alpha)
        # [ v2 ] has dims: (n_b x n x n_inducing x n_inducing)
        v2 = torch.matmul(betat, beta)
        # [ v ] has dims: (n_b x n x n_inducing x n_inducing)
        v = v0 - v1 + v2
        return mu, v


class Sgp(SgpBase):

    def __init__(self,
                 kernel: Kernel,
                 n: int,
                 m: int,
                 z: InducingPoints,
                 sigma: Optional[Tensor] = None):
        """
        __init__ method for Sparse GP Class

        Parameters
        ----------
        kernel : Kernel
            kernel used for sparse GP (e.g., QuadExp)
        n : int
            number of neurons
        m : int
            number of conditions
        z : InducingPoints
            inducing points for sparse GP
        sigma : optional Tensor
            initialization of the output noise standard deviation
            i.e., noise variance is sigma^2
        """
        super().__init__(kernel, n, m, z.n_z, sigma)
        self.z = z

    @property
    def prms(self) -> Tuple[Tensor, Tensor]:
        sigma = softplus(self.sigma)[:, None, None]
        z = self.z.prms
        return sigma, z

    def _expand(self, z: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = z[None, ...]
        x = x[:, None, ...]
        return z, x


class SgpComb(SgpBase):

    def __init__(self,
                 kernel: Combination,
                 n: int,
                 m: int,
                 zs: List[InducingPoints],
                 sigma: Optional[Tensor] = None):
        """
        __init__ method for Sparse GP with Combination Kernels

        Parameters
        ----------
        kernels : Combination Kernel
            combination kernel used for sparse GP (e.g. Product)
        n : int
            number of neurons
        m : int
            number of conditions
        zs : InducingPoints list
            list of inducing points
        sigma : optional Tensor
            initialization of the output noise standard deviation
            i.e., noise variance is sigma^2
        """

        self.zs = zs
        n_inducing = zs[0].n_z
        # check all the us have the same n_inducing
        for z in zs:
            assert (z.n_z == n_inducing)
        super().__init__(kernel, n, m, n_inducing, sigma)

    @property
    def prms(self) -> Tuple[Tensor, Union[List[Tensor], nn.ParameterList]]:
        sigma = softplus(self.sigma)[:, None, None]  # noise
        zs = [z.prms for z in self.zs]  # inducing
        return sigma, zs

    def _expand(self, zs: List[Tensor],
                xs: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        zs = [z[None, ...] for z in zs]
        xs = [x[:, None, ...] for x in xs]
        return zs, xs
