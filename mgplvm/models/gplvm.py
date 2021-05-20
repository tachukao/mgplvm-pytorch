from __future__ import print_function
import numpy as np
from ..utils import softplus
from . import svgp
from .. import rdist, kernels, utils
import torch
from torch import nn, Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import pickle
from .. import lpriors
from ..inducing_variables import InducingPoints
from ..kernels import Kernel
from ..likelihoods import Likelihood
from ..lpriors.common import Lprior
from ..rdist import Rdist


class Gplvm(nn.Module):
    name = "Gplvm"

    def __init__(self, obs, lat_dist: Rdist, lprior: Lprior, n, m, n_samples):
        """
        __init__ method for GPLVM model
        Parameters
        ----------
        obs : Module
            observation model defining p(Y|X)
        lat_dist : Rdist
            variational distirbution q(x)
        lprior : Lprior
            prior p(x) (or null prior if q(x) directly computes KL[q||p])
        n : int
            number of neurons
        m : int
            number of time points / conditions
        n_sample : int
            number of samples/trials
        """
        super().__init__()

        self.obs = obs  #p(Y|X)
        self.svgp = self.obs
        self.n = n
        self.m = m
        self.n_samples = n_samples

        # latent distribution
        self.lat_dist = lat_dist  #Q(X)
        self.lprior = lprior  #P(X)

    def elbo(self,
             data,
             n_mc,
             kmax=5,
             batch_idxs=None,
             sample_idxs=None,
             neuron_idxs=None,
             m=None,
             analytic_kl=False):
        """
        Parameters
        ----------
        data : Tensor
            data with dimensionality (n_samples x n x m)
        n_mc : int
            number of MC samples
        kmax : int
            parameter for estimating entropy for several manifolds
            (not used for some manifolds)
        batch_idxs : Optional int list
            if None then use all data and (batch_size == m)
            otherwise, (batch_size == len(batch_idxs))
        sample_idxs : Optional int list
            if None then use all data 
            otherwise, compute elbo only for selected samples
        neuron_idxs: Optional int list
            if None then use all data 
            otherwise, compute only elbo for selected neurons
        m : Optional int
            used to scale the svgp likelihood and sgp prior.
            If not provided, self.m is used which is provided at initialization.
            This parameter is useful if we subsample data but want to weight the prior as if it was the full dataset.
            We use this e.g. in crossvalidation

        Returns
        -------
        svgp_elbo : Tensor
            evidence lower bound of sparse GP per neuron, batch and sample (n_mc x n)
            note that this is the ELBO for the batch which is proportional to an unbiased estimator for the data.
        kl : Tensor
            estimated KL divergence per batch between variational distribution and prior (n_mc)

        Notes
        -----
        ELBO of the model per batch is [ svgp_elbo - kl ]
        """

        n_samples, n = self.n_samples, self.n
        m = (self.m if m is None else m)

        g, lq = self.lat_dist.sample(torch.Size([n_mc]),
                                     data,
                                     batch_idxs=batch_idxs,
                                     sample_idxs=sample_idxs,
                                     kmax=kmax,
                                     analytic_kl=analytic_kl,
                                     prior=self.lprior)
        # g is shape (n_mc, n_samples, m, d)
        # lq is shape (n_mc x n_samples x m)

        #data = data if sample_idxs is None else data[..., sample_idxs, :, :]
        #data = data if batch_idxs is None else data[..., batch_idxs]

        # note that [ obs.elbo ] recognizes inputs of dims (n_mc x d x m)
        # and so we need to permute [ g ] to have the right dimensions
        #(n_mc x n), (1 x n)
        svgp_lik, svgp_kl = self.obs.elbo(data,
                                          g.transpose(-1, -2),
                                          sample_idxs,
                                          m=m)  #p(Y|g)
        if neuron_idxs is not None:
            svgp_lik = svgp_lik[..., neuron_idxs]
            svgp_kl = svgp_kl[..., neuron_idxs]
        lik = svgp_lik - svgp_kl

        if analytic_kl or ('GP' in self.lat_dist.name):
            #print('analytic KL')
            #kl per MC sample; lq already represents the full KL
            kl = (torch.ones(n_mc).to(data.device)) * lq.sum()
        else:
            # compute kl term for the latents (n_mc, n_samples) per batch
            prior = self.lprior(g, batch_idxs)  #(n_mc)
            #print('prior, lq shapes:', prior.shape, lq.shape)
            kl = lq.sum(-1).sum(-1) - prior  #(n_mc) (sum q(g) over samples, conditions)

        #rescale KL to entire dataset (basically structured conditions)
        batch_size = m if batch_idxs is None else len(batch_idxs)
        sample_size = n_samples if sample_idxs is None else len(sample_idxs)
        kl = (m / batch_size) * (n_samples / sample_size) * kl

        return lik, kl

    def forward(self,
                data,
                n_mc,
                kmax=5,
                batch_idxs=None,
                sample_idxs=None,
                neuron_idxs=None,
                m=None,
                analytic_kl=False):
        """
        Parameters
        ----------
        data : Tensor
            data with dimensionality (n_samples x n x m)
        n_mc : int
            number of MC samples
        kmax : int
            parameter for estimating entropy for several manifolds
            (not used for some manifolds)
        batch_idxs: Optional int list
            if None then use all data and (batch_size == m)
            otherwise, (batch_size == len(batch_idxs))
        sample_idxs : Optional int list
            if None then use all data 
            otherwise, compute elbo only for selected samples
        neuron_idxs: Optional int list
            if None then use all data 
            otherwise, compute only elbo for selected neurons
        m : Optional int
            used to scale the svgp likelihood and sgp prior.
            If not provided, self.m is used which is provided at initialization.
            This parameter is useful if we subsample data but want to weight the prior as if it was the full dataset.
            We use this e.g. in crossvalidation

        Returns
        -------
        elbo : Tensor
            evidence lower bound of the GPLVM model averaged across MC samples and summed over n, m, n_samples (scalar)
        """

        #(n_mc, n), (n_mc)
        lik, kl = self.elbo(data,
                            n_mc,
                            kmax=kmax,
                            batch_idxs=batch_idxs,
                            sample_idxs=sample_idxs,
                            neuron_idxs=neuron_idxs,
                            m=m,
                            analytic_kl=analytic_kl)
        #sum over neurons and mean over  MC samples
        lik = lik.sum(-1).mean()
        kl = kl.mean()

        return lik, kl  #mean across batches, sum across everything else

    def calc_LL(self, data, n_mc, kmax=5, m=None):
        """
        Parameters
        ----------
        data : Tensor
            data with dimensionality (n_samples x n x m)
        n_mc : int
            number of MC samples
        kmax : int
            parameter for estimating entropy for several manifolds
            (not used for some manifolds)
        m : Optional int
            used to scale the svgp likelihood and sgp prior.
            If not provided, self.m is used which is provided at initialization.
            This parameter is useful if we subsample data but want to weight the prior as if it was the full dataset.
            We use this e.g. in crossvalidation

        Returns
        -------
        LL : Tensor
            E_mc[p(Y)] (burda et al.) (scalar)
        """

        #(n_mc, n), (n_mc)
        svgp_elbo, kl = self.elbo(data, n_mc, kmax=kmax, m=m)
        svgp_elbo = svgp_elbo.sum(-1)  #(n_mc)
        LLs = svgp_elbo - kl  # LL for each batch (n_mc)
        assert (LLs.shape == torch.Size([n_mc]))

        LL = (torch.logsumexp(LLs, 0) - np.log(n_mc)) / np.prod(data.shape)

        return LL.detach().cpu()
