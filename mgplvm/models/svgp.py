import abc
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from mgplvm.utils import softplus
from ..base import Module
from ..kernels import Kernel, Combination
from ..inducing_variables import InducingPoints
from typing import Tuple, List, Optional, Union
from torch.distributions import MultivariateNormal, kl_divergence, transform_to, constraints, Normal
from ..likelihoods import Likelihood

jitter: float = 1E-8
log2pi: float = np.log(2 * np.pi)


class SvgpBase(Module, metaclass=abc.ABCMeta):
    def __init__(self,
                 kernel: Kernel,
                 n: int,
                 n_inducing: int,
                 likelihood: Likelihood,
                 q_mu: Optional[Tensor] = None,
                 q_sqrt: Optional[Tensor] = None,
                 whiten=True):
        """
        __init__ method for Base Sparse Variational GP Class
        Parameters
        ----------
        n : int
            number of neurons
        m : int
            number of conditions
        n_inducing : int
            number of inducing points
        q_mu : Optional Tensor
            optional Tensor for initialization
        q_sqrt : Optional Tensor
            optional Tensor for initialization
        whiten : Optional bool
            whiten q if true
        """
        super().__init__()
        self.n = n
        self.n_inducing = n_inducing
        self.kernel = kernel

        q_mu = torch.zeros(n, n_inducing) if q_mu is None else q_mu
        q_sqrt = torch.diag_embed(torch.ones(
            n, n_inducing)) if q_sqrt is None else transform_to(
                constraints.lower_cholesky).inv(q_sqrt)

        self.q_mu = nn.Parameter(q_mu, requires_grad=True)
        self.q_sqrt = nn.Parameter(q_sqrt, requires_grad=True)

        self.likelihood = likelihood
        self.whiten = whiten

    @abc.abstractmethod
    def _expand_z(self, z):
        pass

    @abc.abstractmethod
    def _expand_x(self, x):
        pass

    def prior_kl(self):
        q_mu, q_sqrt, z = self.prms
        z = self._expand_z(z)
        e = torch.eye(self.n_inducing).to(q_mu.device)
        if not self.whiten:
            kzz = self.kernel(z, z) + (e * jitter)
            l = torch.cholesky(kzz, upper=False)
        q = MultivariateNormal(q_mu, scale_tril=q_sqrt)
        p_mu = torch.zeros(self.n, self.n_inducing).to(q_mu.device)
        if not self.whiten:
            prior = MultivariateNormal(p_mu, scale_tril=l)
        else:
            prior = MultivariateNormal(p_mu, scale_tril=e)

        return kl_divergence(q, prior)

    def elbo(self, n_mc: int, y: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        n_mc : int
            number of monte carlo samples
        y : Tensor
            data tensor with dimensions (n_samples x n x m)
        x : Tensor (single kernel) or Tensor list (product kernels)
            input tensor(s) with dimensions (n_mc x n_samples x d x m)

        Returns
        -------
        evidence lower bound : torch.Tensor (n_mc x n)

        Notes
        -----
        Implementation largely follows derivation of the ELBO presented in `here <https://gpflow.readthedocs.io/en/develop/notebooks/theory/SGPR_notes.html>`_.
        """

        kernel = self.kernel
        n_inducing = self.n_inducing  # inducing points
        prior_kl = self.prior_kl()  # prior KL(q(u) || p(u)) (1 x n)
        # predictive mean and var at x
        f_mean, f_var = self.predict(x, full_cov=False)

        #(n_mc, n_samles, n)
        lik = self.likelihood.variational_expectation(y, f_mean, f_var)

        return lik, prior_kl

    def tuning(self, query, n_b=1000, square=False):
        '''
        query is mxd
        return n_b samples from the full model (n_b x n_samples x n x m)
        if square, the outputs are squared (useful e.g. when fitting sqrt spike counts with a Gaussian likelihood)
        '''

        query = torch.unsqueeze(query.T, 0)  #add batch dimension

        mu, v = self.predict(query, False)  #1xn_samplesxnxm, 1xn_samplesxnxm
        # remove batch dimension
        mu = mu[0]  #n x m,
        v = v[0]  # nxm

        #sample from p(f|u)
        dist = Normal(mu, torch.sqrt(v))
        f_samps = dist.sample((n_b, ))  #n_mc x n_samples x n x m

        #sample from observation function p(y|f)
        y_samps = self.likelihood.sample(f_samps)
        #mu, std = y_samps.mean(dim = 0), y_samps.std(dim = 0)

        if square:
            y_samps = y_samps**2

        return y_samps

    def predict(self, x: Tensor, full_cov: bool) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x : Tensor (single kernel) or Tensor list (product kernels)
            test input tensor(s) with dimensions (n_b x n_samples x d x m)
        full_cov : bool
            returns full covariance if true otherwise returns the diagonal

        Returns
        -------
        mu : Tensor 
            mean of predictive density at test inputs [ s ]
        v : Tensor 
            variance/covariance of predictive density at test inputs [ s ]
            if full_cov is true returns full covariance, otherwise
            returns diagonal variance

        Notes
        -----
        """

        q_mu, q_sqrt, z = self.prms
        kernel = self.kernel
        q_mu = q_mu[..., None]

        # see ELBO for explanation of _expand
        z = self._expand_z(z)
        x = self._expand_x(x)
        kzz = kernel(z, z)  # dims: (1 x n x n_z x n_z)
        kzx = kernel(z, x)  # dims: (n_mc x n_samples x n x n_inducing x m)
        e = torch.eye(self.n_inducing,
                      dtype=torch.get_default_dtype()).to(kzz.device)

        # [ l ] has dims: (1 x n x n_inducing x n_inducing)
        l = torch.cholesky(kzz + (jitter * e), upper=False)
        # [ alpha ] has dims: (n_b x n_samples x n x n_inducing x m)
        alpha = torch.triangular_solve(kzx, l, upper=False)[0]
        alphat = alpha.transpose(-1, -2)

        if self.whiten:
            # [ mu ] has dims : (n_b x n_samples x n x m x 1)
            mu = torch.matmul(alphat, q_mu)
        else:
            # [ beta ] has dims : (n_b x n_samples x n x n_inducing x m)
            beta = torch.triangular_solve(alpha,
                                          l.transpose(-1, -2),
                                          upper=True)[0]
            # [ betat ] has dims : (n_b x n_samples x n x m x n_inducing)
            betat = beta.transpose(-1, -2)
            mu = torch.matmul(betat, q_mu)

        if full_cov:
            # [ tmp1 ] has dims : (n_b x n_samples, n x m x n_inducing)
            if self.whiten:
                tmp1 = torch.matmul(alphat, q_sqrt)
            else:
                tmp1 = torch.matmul(betat, q_sqrt)
            # [ v1 ] has dims : (n_b x n_samples x n x m x m)
            v1 = torch.matmul(tmp1, tmp1.transpose(-1, -2))
            # [ v2 ] has dims : (n_b x n_samples x n x m x m)
            v2 = torch.matmul(alphat, alphat)
            # [ kxx ] has dims : (n_b x n_samples x n x m x m)
            kxx = kernel(x, x)
            v = kxx + v1 - v2
        else:
            # [ kxx ] has dims : (n_b x n_samples x n x m)
            kxx = kernel.diagK(x)
            # [ tmp1 ] has dims : (n_b x n_samples x n x m x n_inducing)
            if self.whiten:
                tmp1 = torch.matmul(alphat, q_sqrt)
            else:
                tmp1 = torch.matmul(betat, q_sqrt)
            # [ v1 ] has dims : (n_b x n_samples x n x m)
            v1 = torch.square(tmp1).sum(-1)
            # [ v2 ] has dims : (n_b x n_samples x n x m)
            v2 = torch.square(alpha).sum(-2)
            v = kxx + v1 - v2

        return mu.squeeze(-1), v


class Svgp(SvgpBase):
    def __init__(self,
                 kernel: Kernel,
                 n: int,
                 z: InducingPoints,
                 likelihood: Likelihood,
                 whiten: Optional[bool] = True):
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
        likelihood : Likelihood
            likleihood p(y | f) 
        whiten : Optional bool
            whiten q if true

        Returns
        -------
        
        """
        # initalize q_sqrt^2 at the prior kzz
        n_inducing = z.n_z
        _z = self._expand_z(z.prms)
        e = torch.eye(n_inducing)
        kzz = kernel(_z, _z) + (e * jitter)
        l = torch.cholesky(kzz, upper=False)
        super().__init__(kernel,
                         n,
                         n_inducing,
                         likelihood,
                         q_sqrt=l,
                         whiten=whiten)
        self.z = z

    @property
    def prms(self) -> Tuple[Tensor, Tensor, Tensor]:
        z = self.z.prms
        q_mu = self.q_mu
        q_sqrt = transform_to(constraints.lower_cholesky)(self.q_sqrt)
        return q_mu, q_sqrt, z

    def _expand_z(self, z: Tensor) -> Tensor:
        z = z[None, ...]
        return z

    def _expand_x(self, x: Tensor) -> Tensor:
        x = x[:, :, None, ...]
        return x


#class SvgpComb(SvgpBase):
#    def __init__(self,
#                 kernel: Combination,
#                 n: int,
#                 zs: List[InducingPoints],
#                 likelihood: Likelihood,
#                 whiten: Optional[bool] = True):
#        """
#        __init__ method for Sparse GP with Combination Kernels
#        Parameters
#        ----------
#        kernels : Combination Kernel
#            combination kernel used for sparse GP (e.g., Product)
#        n : int
#            number of neurons
#        m : int
#            number of conditions
#        zs : InducingPoints list
#            list of inducing points
#        likleihood: Likelihood
#            likelihood p(y|f)
#        whiten : Optional bool
#            whiten q if true
#        """
#
#        n_inducing = zs[0].n_inducing
#        # check all the zs have the same n_inducing
#        for z in zs:
#            assert (z.n_inducing == n_inducing)
#
#        # initialize q_sqrt
#        _zs = [z.prms for z in zs]
#        _z = self._expand_z(_zs)
#        e = torch.eye(n_inducing)
#        kzz = kernel(z, z) + (e * jitter)
#        l = torch.cholesky(kzz, upper=False)
#        super().__init__(kernel,
#                         n,
#                         n_inducing,
#                         likelihood,
#                         q_sqrt=l,
#                         whiten=whiten)
#        self.zs = zs
#
#    @property
#    def prms(self) -> Tuple[Tensor, Union[List[Tensor], nn.ParameterList]]:
#        zs = [z.prms for z in self.zs]
#        q_mu = self.q_mu
#        q_sqrt = torch.distributions.transform_to(
#            MultivariateNormal.arg_constraints['scale_tril'])(self.q_sqrt)
#        return q_mu, q_sqrt, zs
#
#    def _expand_z(self, zs: List[Tensor]) -> Tuple[List[Tensor]]:
#        zs = [z[None, ...] for z in zs]
#        return zs
#
#    def _expand_x(self, xs: List[Tensor]) -> Tuple[List[Tensor]]:
#        xs = [x[:, None, ...] for x in xs]
#        return xs
#
