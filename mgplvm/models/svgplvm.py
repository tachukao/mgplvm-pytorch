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


class SvgpLvm(nn.Module):
    name = "Svgplvm"

    def __init__(self,
                 n: int,
                 m: int,
                 n_samples: int,
                 z: InducingPoints,
                 kernel: Kernel,
                 likelihood: Likelihood,
                 lat_dist: Rdist,
                 lprior=Lprior,
                 whiten: bool = True,
                 tied_samples=True):
        """
        __init__ method for Vanilla model
        Parameters
        ----------
        n : int
            number of neurons
        m : int
            number of conditions
        n_samples: int
            number of samples
        z : Inducing Points
            inducing points
        kernel : Kernel
            kernel used for GP regression
        likelihood : Likelihood
            likelihood p(y|f)
        lat_dist : rdist
            latent distribution
        lprior: Lprior
            log prior over the latents
        whiten: bool
            parameter passed to Svgp
        """
        super().__init__()
        self.n = n
        self.m = m
        self.n_samples = n_samples
        self.kernel = kernel
        self.z = z
        self.likelihood = likelihood
        self.whiten = whiten
        self.svgp = svgp.Svgp(self.kernel,
                              n,
                              m,
                              n_samples,
                              self.z,
                              likelihood,
                              whiten=whiten,
                              tied_samples=tied_samples)
        # latent distribution
        self.lat_dist = lat_dist
        self.lprior = lprior

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
        # g is shape (n_samples, n_mc, m, d)
        # lq is shape (n_mc x n_samples x m)

        #data = data if sample_idxs is None else data[..., sample_idxs, :, :]
        #data = data if batch_idxs is None else data[..., batch_idxs]

        # note that [ svgp.elbo ] recognizes inputs of dims (n_mc x d x m)
        # and so we need to permute [ g ] to have the right dimensions
        #(n_mc x n), (1 x n)
        svgp_lik, svgp_kl = self.svgp.elbo(data,
                                           g.transpose(-1, -2),
                                           sample_idxs,
                                           m=m)
        if neuron_idxs is not None:
            svgp_lik = svgp_lik[..., neuron_idxs]
            svgp_kl = svgp_kl[..., neuron_idxs]

        batch_size = m if batch_idxs is None else len(batch_idxs)
        sample_size = n_samples if sample_idxs is None else len(sample_idxs)
        lik = svgp_lik - svgp_kl

        if analytic_kl:
            kl = (torch.ones(n_mc).to(data.device)) * lq.sum(
            )  #kl per MC sample; lq already represents the full KL
        else:
            # compute kl term for the latents (n_mc, n_samples) per batch
            prior = self.lprior(g, batch_idxs)  #(n_mc)
            #print('prior, lq shapes:', prior.shape, lq.shape)
            kl = lq.sum(-1).sum(
                -1) - prior  #(n_mc) (sum q(g) over samples, conditions)

        #rescale KL to entire dataset (basically structured conditions)
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
