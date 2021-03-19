# Bayesian Factor Analaysis
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
from torch.distributions import MultivariateNormal, LowRankMultivariateNormal, kl_divergence, transform_to, constraints, Normal
from ..likelihoods import Likelihood

jitter: float = 1E-8
log2pi: float = np.log(2 * np.pi)


class Bfa(Module):
    """
    Bayesian Factor Analysis
    Assumes Gaussian observation noise
    Computes log_prob and posterior predictions exactly
    """

    def __init__(self,
                 n: int,
                 sigma: Optional[Tensor] = None,
                 learn_sigma=True):
        super().__init__()
        if sigma is None:
            sigma = torch.ones(n,)  # TODO: FA init
        self._sigma = nn.Parameter(data=sigma, requires_grad=learn_sigma)
        self.n = n

    @property
    def prms(self) -> Tensor:
        """p(y_i | f_i) = N(0, sigma^2)"""
        variance = torch.square(self._sigma)
        return variance

    @property
    def sigma(self) -> Tensor:
        return (1e-20 + self.prms).sqrt()

    def dist(self, x):
        """
        construct low rank prior MVN = N(0, X^T X)
        """
        m = x.shape[-1]
        d = x.shape[-2]
        cov_factor = x[..., None, :, :].transpose(-1, -2)  #(..., mxd)
        cov_diag = self.prms[:, None] * torch.ones(m)
        dist = LowRankMultivariateNormal(loc=torch.zeros(self.n, m),
                                         cov_factor=cov_factor,
                                         cov_diag=cov_diag)
        return dist

    def log_prob(self, y, x):
        """compute prior p(y) = N(y|0, X^T X)"""
        dist = self.dist(x)
        lp = dist.log_prob(y)
        return lp.sum()

    def predict(self, xstar, y, x, full_cov=False):
        """
        compute posterior p(f* | x, y)
        """
        m = x.shape[-1]
        dist = self.dist(x)
        prec = dist.precision_matrix  #(K+sigma^2I)^-1
        l = torch.cholesky(prec, upper=False)  #mxm??
        x = x[..., None, :, :]
        xl = x.matmul(l)
        _mu = xl.matmul(l.transpose(-1, -2)).matmul(y[..., None]).squeeze(-1)
        mu = _mu.matmul(xstar)
        xstar = xstar[..., None, :, :]
        if not full_cov:
            return mu, torch.square(xstar).sum(-2) - torch.square(
                xstar.transpose(-1, -2).matmul(xl)).sum(-1)
        else:
            z = torch.eye(m) - xl.matmul(xl.transpose(-1, 2))
            return mu, xstar.transpose(-1, -2).matmul(z).matmul(xstar)


class Bvfa(Module):

    def __init__(self,
                 n: int,
                 d: int,
                 m: int,
                 n_samples: int,
                 likelihood: Likelihood,
                 q_mu: Optional[Tensor] = None,
                 q_sqrt: Optional[Tensor] = None,
                 tied_samples=True):
        """
        __init__ method for Base Variational Factor Analysis 
        Parameters
        ----------
        n : int
            number of neurons
        d: int
            latent dimensionality
        m : int 
            number of conditions
        n_samples : int 
            number of samples
        likelihood : Likelihood
            likliehood module used for computing variational expectation
        q_mu : Optional Tensor
            optional Tensor for initialization
        q_sqrt : Optional Tensor
            optional Tensor for initialization
        tied_samples : Optional bool
        """
        super().__init__()
        self.n = n
        self.d = d
        self.m = m
        self.tied_samples = tied_samples
        self.n_samples = n_samples

        if q_mu is None:
            if tied_samples:
                q_mu = torch.zeros(1, n, d)
            else:
                q_mu = torch.zeros(n_samples, n, d)

        if q_sqrt is None:
            if tied_samples:
                q_sqrt = torch.diag_embed(torch.ones(1, n, d))
            else:
                q_sqrt = torch.diag_embed(torch.ones(n_samples, n, d))
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

    def prior_kl(self, sample_idxs=None):
        """
        KL(p(f) || q(f))
        """
        q_mu, q_sqrt = self.prms
        assert (q_mu.shape[0] == q_sqrt.shape[0])
        if not self.tied_samples and sample_idxs is not None:
            q_mu = q_mu[sample_idxs]
            q_sqrt = q_sqrt[sample_idxs]
        q = MultivariateNormal(q_mu, scale_tril=q_sqrt)
        e = torch.eye(self.d).to(q_mu.device)
        p_mu = torch.zeros(self.n, self.d).to(q_mu.device)
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
        """

        assert (x.shape[-3] == y.shape[-3])
        assert (x.shape[-1] == y.shape[-1])
        batch_size = x.shape[-1]
        sample_size = x.shape[-3]

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
        lik = lik.sum(-2)
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

        """

        q_mu, q_sqrt = self.prms

        assert (q_mu.shape[0] == q_sqrt.shape[0])
        if (not self.tied_samples) and sample_idxs is not None:
            q_mu = q_mu[sample_idxs]
            q_sqrt = q_sqrt[sample_idxs]

        mu = q_mu.matmul(x)  # n_b x n_samples x n x m
        l = x[..., None, :, :].transpose(-1, -2).matmul(
            q_sqrt)  # n_b x n_samples x m x d
        if not full_cov:
            return mu, torch.square(l).sum(-1)
        else:
            return mu, l.matmul(l.transpose(-1, -2))

    @property
    def prms(self) -> Tuple[Tensor, Tensor]:
        q_mu = self.q_mu
        q_sqrt = transform_to(constraints.lower_cholesky)(self.q_sqrt)
        return q_mu, q_sqrt
