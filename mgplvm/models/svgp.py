import abc
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from mgplvm.utils import softplus
from ..base import Module
from ..kernels import Kernel
from ..inducing_variables import InducingPoints
from typing import Tuple, List, Optional, Union
from torch.distributions import MultivariateNormal, kl_divergence, transform_to, constraints, Normal
from ..likelihoods import Likelihood
from .gp_base import GpBase
import itertools

jitter: float = 1E-8
log2pi: float = np.log(2 * np.pi)


class SvgpBase(GpBase):

    def __init__(self,
                 kernel: Kernel,
                 n: int,
                 m: int,
                 n_samples: int,
                 n_inducing: int,
                 likelihood: Likelihood,
                 q_mu: Optional[Tensor] = None,
                 q_sqrt: Optional[Tensor] = None,
                 whiten=True,
                 tied_samples=True):
        """
        __init__ method for Base Sparse Variational GP Class (p(Y|X))
        Parameters
        ----------
        n : int
            number of neurons
        m : int 
            number of conditions
        n_samples : int 
            number of samples
        n_inducing : int
            number of inducing points
        likelihood : Likelihood
            likliehood module used for computing variational expectation
        q_mu : Optional Tensor
            optional Tensor for initialization
        q_sqrt : Optional Tensor
            optional Tensor for initialization
        whiten : Optional bool
            whiten q if true
        tied_samples : Optional bool
        """
        super().__init__()
        self.n = n
        self.m = m
        self.n_inducing = n_inducing
        self.tied_samples = tied_samples
        self.n_samples = n_samples
        self.kernel = kernel

        if q_mu is None:
            if tied_samples:
                q_mu = torch.zeros(1, n, n_inducing)
            else:
                q_mu = torch.zeros(n_samples, n, n_inducing)

        if q_sqrt is None:
            if tied_samples:
                q_sqrt = torch.diag_embed(torch.ones(1, n, n_inducing))
            else:
                q_sqrt = torch.diag_embed(torch.ones(n_samples, n, n_inducing))
        else:
            q_sqrt = transform_to(constraints.lower_cholesky).inv(q_sqrt)

        assert (q_mu is not None)
        assert (q_sqrt is not None)
        if self.tied_samples:
            assert (q_mu.shape[0] == 1)
            assert (q_sqrt.shape[0] == 1)
        else:
            assert (q_mu.shape[0] == n_samples)
            assert (q_sqrt.shape[0] == n_samples)

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

    def prior_kl(self, sample_idxs=None):
        q_mu, q_sqrt, z = self.prms
        assert (q_mu.shape[0] == q_sqrt.shape[0])
        if not self.tied_samples and sample_idxs is not None:
            q_mu = q_mu[sample_idxs]
            q_sqrt = q_sqrt[sample_idxs]
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

    def elbo(self,
             y: Tensor,
             x: Tensor,
             sample_idxs: Optional[List[int]] = None,
             m: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        y : Tensor
            data tensor with dimensions (n_samples x n x m)
        x : Tensor (single kernel) or Tensor list (product kernels)
            input tensor(s) with dimensions (n_mc x n_samples x d x m)
        m : Optional int
            used to scale the svgp likelihood.
            If not provided, self.m is used which is provided at initialization.
            This parameter is useful if we subsample data but want to weight the prior as if it was the full dataset.
            We use this e.g. in crossvalidation

        Returns
        -------
        lik, prior_kl : Tuple[torch.Tensor, torch.Tensor]
            lik has dimensions (n_mc x n) 
            prior_kl has dimensions (n)

        Notes
        -----
        Implementation largely follows derivation of the ELBO presented in `here <https://gpflow.readthedocs.io/en/develop/notebooks/theory/SGPR_notes.html>`_.
        """

        assert (x.shape[-3] == y.shape[-3]) #Trials
        assert (x.shape[-1] == y.shape[-1]) #Time
        batch_size = x.shape[-1]
        sample_size = x.shape[-3]

        kernel = self.kernel
        n_inducing = self.n_inducing  # inducing points

        # prior KL(q(u) || p(u)) (1 x n) if tied_samples otherwise (n_samples x n)
        prior_kl = self.prior_kl(sample_idxs)
        # predictive mean and var at x
        f_mean, f_var = self.predict(x, full_cov=False, sample_idxs=sample_idxs)
        prior_kl = prior_kl.sum(-2)
        if not self.tied_samples:
            prior_kl = prior_kl * (self.n_samples / sample_size)

        #(n_mc, n_samles, n)
        lik = self.likelihood.variational_expectation(y, f_mean, f_var)
        # scale is (m / batch_size) * (self.n_samples / sample size)
        # to compute an unbiased estimate of the likelihood of the full dataset
        m = (self.m if m is None else m)
        scale = (m / batch_size) * (self.n_samples / sample_size)
        lik = lik.sum(-2)  #sum over samples
        lik = lik * scale

        return lik, prior_kl

    def sample(self,
               query: Tensor,
               n_mc: int = 1000,
               square: bool = False,
               noise: bool = True):
        """
        Parameters
        ----------
        query : Tensor (single kernel)
            test input tensor with dimensions (n_samples x d x m)
        n_mc : int
            numper of samples to return
        square : bool
            determines whether to square the output
        noise : bool
            determines whether we also sample explicitly from the noise model or simply return samples of the mean

        Returns
        -------
        y_samps : Tensor
            samples from the model (n_mc x n_samples x d x m)
        """

        query = query[None, ...]  #add batch dimension (1 x n_samples x d x m)

        mu, v = self.predict(query, False)  #1xn_samplesxnxm, 1xn_samplesxnxm
        # remove batch dimension
        mu = mu[0]  #n_samples x n x m,
        v = v[0]  # n_samples x n x m

        #sample from p(f|u)
        dist = Normal(mu, torch.sqrt(v))

        f_samps = dist.sample((n_mc,))  #n_mc x n_samples x n x m

        if noise:
            #sample from observation function p(y|f)
            y_samps = self.likelihood.sample(f_samps)  #n_mc x n_samples x n x m
        else:
            #compute mean observations mu(f) for each f
            y_samps = self.likelihood.dist_mean(
                f_samps)  #n_mc x n_samples x n x m

        if square:
            y_samps = y_samps**2

        return y_samps

    def predict(self,
                x: Tensor,
                full_cov: bool,
                sample_idxs=None) -> Tuple[Tensor, Tensor]:
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

        assert (q_mu.shape[0] == q_sqrt.shape[0])
        if (not self.tied_samples) and sample_idxs is not None:
            q_mu = q_mu[sample_idxs]
            q_sqrt = q_sqrt[sample_idxs]

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

    name = "Svgp"

    def __init__(self,
                 kernel: Kernel,
                 n: int,
                 m: int,
                 n_samples: int,
                 z: InducingPoints,
                 likelihood: Likelihood,
                 whiten: Optional[bool] = True,
                 tied_samples: Optional[bool] = True):
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
        n_samples : int
            number of samples 
        z : InducingPoints
            inducing points for sparse GP
        likelihood : Likelihood
            likleihood p(y | f) 
        whiten : Optional bool
            whiten q if true
        tied_samples : Optional bool

        Returns
        -------
        
        """
        # initalize q_sqrt^2 at the prior kzz
        n_inducing = z.n_z
        _z = self._expand_z(z.prms)
        e = torch.eye(n_inducing)
        kzz = kernel(_z, _z) + (e * jitter)
        l = torch.cholesky(kzz, upper=False)[None, ...]

        super().__init__(kernel,
                         n,
                         m,
                         n_samples,
                         n_inducing,
                         likelihood,
                         whiten=whiten,
                         tied_samples=tied_samples)
        self.z = z

    @property
    def prms(self) -> Tuple[Tensor, Tensor, Tensor]:
        z = self.z.prms
        q_mu = self.q_mu
        q_sqrt = transform_to(constraints.lower_cholesky)(self.q_sqrt)
        return q_mu, q_sqrt, z

    def _expand_z(self, z: Tensor) -> Tensor:
        return z

    def _expand_x(self, x: Tensor) -> Tensor:
        x = x[..., None, :, :]
        return x

    @property
    def msg(self):
        return self.kernel.msg + self.likelihood.msg

    def g0_parameters(self):
        return [self.q_mu, self.q_sqrt]

    def g1_parameters(self):
        return list(
            itertools.chain.from_iterable([
                self.kernel.parameters(),
                self.z.parameters(),
                self.likelihood.parameters()
            ]))
