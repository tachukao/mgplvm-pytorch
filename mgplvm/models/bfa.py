# Bayesian Factor Analaysis
import abc
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from mgplvm.utils import softplus, inv_softplus
from ..base import Module
from ..kernels import Kernel
from ..inducing_variables import InducingPoints
from typing import Tuple, List, Optional, Union
from torch.distributions import MultivariateNormal, LowRankMultivariateNormal, kl_divergence, transform_to, constraints, Normal
from ..likelihoods import Likelihood
from sklearn import decomposition
from .gp_base import GpBase
import itertools

jitter: float = 1E-8
log2pi: float = np.log(2 * np.pi)


def batch_capacitance_tril(W, D):
    r"""
    Copied from pytorch source code
    Computes Cholesky of :math:`I + W.T @ inv(D) @ W` for a batch of matrices :math:`W`
    and a batch of vectors :math:`D`.
    """
    m = W.size(-1)
    Wt_Dinv = W.transpose(-1, -2) / D.unsqueeze(-2)
    K = torch.matmul(Wt_Dinv, W).contiguous()
    K.view(-1, m * m)[:, ::m + 1] += 1
    return torch.cholesky(K)


class Bfa(GpBase):
    """
    Bayesian Factor Analysis
    Assumes Gaussian observation noise
    Computes log_prob and posterior predictions exactly
    """

    name = "Bfa"

    def __init__(self,
                 n: int,
                 d: int,
                 sigma: Optional[Tensor] = None,
                 learn_sigma=True,
                 Y=None,
                 learn_neuron_scale=False,
                 ard=False,
                 learn_scale=None):

        super().__init__()

        if Y is not None:
            n_samples_fa, n_fa, m_fa = Y.shape
            mod = decomposition.FactorAnalysis(n_components=d)
            Y_fa = Y.transpose(0, 2, 1).reshape(n_samples_fa * m_fa, n_fa)
            mudata = mod.fit_transform(Y_fa)  #m*n_samples x d
            C = torch.tensor(mod.components_.T)  # (n x d)

        #### initialize noise parameters ####
        if sigma is None:
            if Y is None:
                sigma = torch.ones(n,)  # TODO: FA init
            else:
                sigma = torch.tensor(np.sqrt(mod.noise_variance_))

        self._sigma = nn.Parameter(data=sigma, requires_grad=learn_sigma)
        self.n = n

        #### initialize prior parameters ####

        _scale = torch.ones(1)
        _dim_scale = torch.ones(d)
        _neuron_scale = torch.ones(n)
        if learn_scale is None:
            learn_scale = not (ard or learn_neuron_scale)

        if Y is not None:  #initialize from FA
            if learn_scale:
                _scale = torch.square(C).mean().sqrt()  #global scale
            if learn_neuron_scale:
                _neuron_scale = torch.square(C).mean(1).sqrt()  #per neuron
            if ard:
                _dim_scale = torch.square(C).mean(0).sqrt()  #per latent

        self._scale = nn.Parameter(inv_softplus(_scale),
                                   requires_grad=learn_scale)
        self._neuron_scale = nn.Parameter(inv_softplus(_neuron_scale),
                                          requires_grad=learn_neuron_scale)
        self._dim_scale = nn.Parameter(inv_softplus(_dim_scale),
                                       requires_grad=ard)

    @property
    def prms(self) -> Tensor:
        """p(y_i | f_i) = N(0, sigma^2)"""
        variance = torch.square(self._sigma)
        return variance

    @property
    def sigma(self) -> Tensor:
        return (1e-20 + self.prms).sqrt()

    @property
    def scale(self):
        return softplus(self._scale)

    @property
    def neuron_scale(self):
        return softplus(self._neuron_scale)[:, None]

    @property
    def dim_scale(self):
        return softplus(self._dim_scale)[:, None]

    def _dist(self, x):
        """
        construct low rank prior MVN = N(0, X^T X + sigma^2 I)
        """
        m = x.shape[-1]
        x = self.scale * self.dim_scale * x
        cov_factor = x[..., None, :, :].transpose(-1,
                                                  -2)  #(n_samples x 1 x m x d)
        cov_factor = self.neuron_scale[
            ..., None] * cov_factor  #(n_samples x n x m x d)
        cov_diag = self.prms[:, None] * torch.ones(m).to(x.device)  #(n x m)

        dist = LowRankMultivariateNormal(loc=torch.zeros(self.n,
                                                         m).to(x.device),
                                         cov_factor=cov_factor,
                                         cov_diag=cov_diag)
        return dist

    def log_prob(self, y, x):
        """compute prior p(y) = N(y|0, X^T X)"""
        lp = self._dist(x).log_prob(y)  #(n_mc x n_samples x n)
        return lp

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

        Returns
        -------
        lik, prior_kl : Tuple[torch.Tensor, torch.Tensor]
            lik has dimensions (n_mc x n) 
            prior_kl has dimensions (n) and is zero
        """

        lik = self.log_prob(y, x)  #( (n_mc) x n_samples x n)
        lik = lik.sum(-2)
        prior_kl = torch.zeros(self.n).to(x.device)
        return lik, prior_kl

    def predict(self, xstar, y, x, full_cov=False):
        """
        compute posterior p(f* | x, y)
        """

        prec = self._dist(x.squeeze()).precision_matrix  #(K+sigma^2I)^-1
        m = x.shape[-1]
        d = x.shape[-2]

        x = self.scale * self.dim_scale * x
        xstar = self.scale * self.dim_scale * xstar

        x = x[..., None, :, :]  #(...,d,m)
        xstar = xstar[..., None, :, :]  #(...,d,m)
        xt = x.transpose(-1, -2)  #(...,m,d)
        variance = self.prms  #(n) (p(Y|F) variance)
        cov_diag = variance[..., None] * torch.ones(m)  #(n x m)
        capacitance_tril = batch_capacitance_tril(xt, cov_diag)

        xdinv = (x / variance[..., None, None])
        A = torch.triangular_solve(xdinv, capacitance_tril,
                                   upper=False)[0]  #(...,d,m)
        y = y[..., None]
        _mu1 = xdinv.matmul(y)
        _mu2 = x.matmul(A.transpose(-1, -2).matmul(A.matmul(y)))
        mu = xstar.transpose(-1, -2).matmul(_mu1 - _mu2).squeeze(-1)
        if not full_cov:
            v1 = torch.square(xstar).sum(-2)
            v2 = xstar.transpose(-1, -2).matmul(
                x / variance[:, None, None].sqrt()).square().sum(-1)
            v3 = A.matmul(x.transpose(-1, -2)).matmul(xstar).square().sum(-2)
            v = v1 - v2 + v3
            return mu, v1 - v2 + v3
        else:
            xxT = xdinv.matmul(x.transpose(-1, -2))
            xAT = x.matmul(A.transpose(-1, -2))
            xATAxT = xAT.matmul(xAT.transpose(-1, -2))
            z = torch.eye(d) - xxT + xATAxT
            c = xstar.transpose(-1, -2).matmul(z).matmul(xstar)
            return mu, c

    def g0_parameters(self):
        return []

    def g1_parameters(self):
        return [self._sigma, self._neuron_scale, self._dim_scale, self._scale]

    @property
    def msg(self):
        return ('scale {:.3f} |').format(
            (self.scale.mean() * self.neuron_scale.mean() *
             self.dim_scale.mean()).item())


class Bvfa(GpBase):
    name = "Bvfa"

    def __init__(self,
                 n: int,
                 d: int,
                 m: int,
                 n_samples: int,
                 likelihood: Likelihood,
                 q_mu: Optional[Tensor] = None,
                 q_sqrt: Optional[Tensor] = None,
                 tied_samples=True,
                 Y=None,
                 learn_neuron_scale=False,
                 ard=False,
                 learn_scale=None,
                rel_scale = 1):
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
        #self.z, self.kernel = [NoneClass() for i in range(2)]

        #### initialize prior parameters ####

        _scale = torch.ones(1)
        _dim_scale = torch.ones(d)
        _neuron_scale = torch.ones(n)
        if learn_scale is None:
            learn_scale = not (ard or learn_neuron_scale)

        if Y is not None:  #initialize from FA
            n_samples_fa, n_fa, m_fa = Y.shape
            mod = decomposition.FactorAnalysis(n_components=d)
            Y_fa = Y.transpose(0, 2, 1).reshape(n_samples_fa * m_fa, n_fa)
            mudata = mod.fit_transform(Y_fa)  #m*n_samples x d
            C = torch.tensor(mod.components_.T)  # (n x d)
            #print(C.shape)
            if learn_scale:
                _scale = rel_scale*torch.square(C).mean().sqrt()  #global scale
            if learn_neuron_scale:
                _neuron_scale = rel_scale*torch.square(C).mean(1).sqrt()  #per neuron
            if ard:
                _dim_scale = rel_scale*torch.square(C).mean(0).sqrt()  #per latent

        self._scale = nn.Parameter(inv_softplus(_scale),
                                   requires_grad=learn_scale)
        self._neuron_scale = nn.Parameter(inv_softplus(_neuron_scale),
                                          requires_grad=learn_neuron_scale)
        self._dim_scale = nn.Parameter(inv_softplus(_dim_scale),
                                       requires_grad=ard)

        #### initialize variational distribution (should we initialize this to the Gaussian ground truth?)####
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

    @property
    def scale(self):
        return softplus(self._scale)

    @property
    def neuron_scale(self):
        return softplus(self._neuron_scale)[:, None]

    @property
    def dim_scale(self):
        return softplus(self._dim_scale)[:, None]

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
        return kl_divergence(q, prior) ##consider implementing this directly

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

        x = self.scale * self.dim_scale * x  #multiply each dimension by the prior scale

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

        #multiply the posterior by a scale factor for each neuron
        q_mu, q_sqrt = self.neuron_scale * q_mu, self.neuron_scale[
            ..., None] * q_sqrt
        return q_mu, q_sqrt

    def g0_parameters(self):
        return [self.q_mu, self.q_sqrt]

    def g1_parameters(self):
        return list(
            itertools.chain.from_iterable([
                self.likelihood.parameters(),
                [self._scale, self._neuron_scale, self._dim_scale]
            ]))

    @property
    def msg(self):
        newmsg = ('scale {:.3f} |').format(
            (self.scale.mean() * self.neuron_scale.mean() *
             self.dim_scale.mean()).item())
        return newmsg + self.likelihood.msg


class Fa(GpBase):
    """
    Standard non-Bayesian Factor Analysis
    Assumes Gaussian observation noise
    Computes log_prob and posterior predictions exactly
    """

    name = "Fa"

    def __init__(self,
                 n: int,
                 d: int,
                 sigma: Optional[Tensor] = None,
                 learn_sigma=True,
                 Y=None):
        """
        n: number of neurons
        d: number of latents
        """
        super().__init__()
        self.n = n

        if Y is None:
            C = torch.randn(n, d) * d**(-0.5)  # TODO: FA init
            sigma = torch.ones(n,) * 0.5  # TODO: FA init
        else:
            n_samples, n, m = Y.shape
            mod = decomposition.FactorAnalysis(n_components=d)
            Y = Y.transpose(0, 2, 1).reshape(n_samples * m, n)
            mudata = mod.fit_transform(Y)  #m*n_samples x d
            sigma = torch.tensor(np.sqrt(mod.noise_variance_))
            C = torch.tensor(mod.components_.T)

        self._sigma = nn.Parameter(data=sigma, requires_grad=learn_sigma)
        self.C = nn.Parameter(data=C, requires_grad=True)

        #self.likelihood, self.z, self.kernel = [NoneClass() for i in range(3)]

    @property
    def prms(self) -> Tensor:
        """p(y_i | f_i) = N(0, sigma^2_i)"""
        variance = torch.square(self._sigma)
        return variance

    @property
    def sigma(self) -> Tensor:
        return (1e-20 + self.prms).sqrt()

    def log_prob(self, y, x):
        """
        compute p(y|X) = N(y|CX, I)
        x is (n_mc x n_samples x d x m)
        y is (n_samples x n x m)
        """
        mean = self.C @ x  #(... x n x m)
        mean = mean.transpose(-1, -2)  #(... x m x n)
        dist = Normal(loc=mean, scale=self.sigma)
        lp = dist.log_prob(y.transpose(-1, -2))  #(... x m x n)
        #print('lp:', lp.shape)
        return lp

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

        Returns
        -------
        lik, prior_kl : Tuple[torch.Tensor, torch.Tensor]
            lik has dimensions (n_mc x n) 
            prior_kl has dimensions (n) and is zero
        """

        lik = self.log_prob(y, x)  #(n_mc x n_samples x m x n)
        lik = lik.sum(-2).sum(-2)  #n_mc x n
        prior_kl = torch.zeros(self.n).to(x.device)
        return lik, prior_kl

    def predict(self, xstar, full_cov=False):
        """
        compute posterior p(f* | x, y, C) = N(C@x*, Sig)
        """
        mu = self.C @ xstar  #(n_samples x n x m)
        cov = torch.zeros(mu.shape)  #p(f|C, x) is a delta function
        if not full_cov:
            return mu, cov
        else:
            return mu, torch.diag_embed(cov)

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

        mu, _ = self.predict(query, False)  #1xn_samplesxnxm, 1xn_samplesxnxm
        # remove batch dimension
        mu = mu[0]  #n_samples x n x m,
        #sample from p(f|x) which is a delta function for FA
        f_samps = mu

        if noise:
            #sample from observation function p(y|f)
            dist = Normal(loc=f_samps, scale=self.sigma[..., None])
            y_samps = dist.sample(n_mc)  #n_mc x n_samples x n x m
        else:
            #compute mean observations mu(f) for each f
            y_samps = torch.ones(
                n_mc, mu.shape[0], mu.shape[1], mu.shape[2]).to(
                    query.device) * f_samps  #n_mc x n_samples x n x m

        if square:
            y_samps = y_samps**2

        return y_samps

    def g0_parameters(self):
        return []

    def g1_parameters(self):
        return [self._sigma, self.C]

    @property
    def msg(self):
        return ''
