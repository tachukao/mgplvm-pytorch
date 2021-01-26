from __future__ import print_function
import numpy as np
from ..utils import softplus
from . import sgp, svgp
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
                 z: InducingPoints,
                 kernel: Kernel,
                 likelihood: Likelihood,
                 lat_dist: Rdist,
                 lprior=Lprior,
                 whiten: bool = True):
        """
        __init__ method for Vanilla model
        Parameters
        ----------
        n : int
            number of neurons
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
        self.kernel = kernel
        self.z = z
        self.likelihood = likelihood
        self.whiten = whiten
        self.svgp = svgp.Svgp(self.kernel,
                              n,
                              self.z,
                              likelihood,
                              whiten=whiten)
        # latent distribution
        self.lat_dist = lat_dist
        self.lprior = lprior

    def elbo(self,
             data,
             n_mc,
             kmax=5,
             batch_idxs=None,
             ts=None,
             neuron_idxs=None):
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

        n_samples, n, m = data.shape
        
        g, lq = self.lat_dist.sample(torch.Size([n_mc]), data, batch_idxs)
        # g is shape (n_samples, n_mc, m, d)
        # lq is shape (n_mc x n_samples x m)

        data = data if batch_idxs is None else data[:, :, batch_idxs]
        ts = ts if (ts is None or batch_idxs is None) else ts[batch_idxs]
        batch_size = m if batch_idxs is None else len(batch_idxs) #batch size

        # note that [ svgp.elbo ] recognizes inputs of dims (n_mc x d x m)
        # and so we need to permute [ g ] to have the right dimensions
        #scale the svgp prior by batch_size/m for unbiased estimator
        #(n_mc x n)
        svgp_elbo = self.svgp.elbo(n_mc, data, g.transpose(-1, -2), scale = batch_size/m)
        if neuron_idxs is not None:
            svgp_elbo = svgp_elbo[..., neuron_idxs]
            
        # compute kl term for the latents (n_mc, n_samples)
        prior = self.lprior(g, ts)  #(n_mc, n_samples)
        kl = lq.sum(-1) - prior  #(n_mc, n_samples) (sum q(g) over conditions)
        kl = kl.sum(-1) #sum prior KL over samples (basically structured conditions)

        #lik, svgp_kl = self.svgp.elbo(n_mc, data, g.transpose(-1, -2))
        #ELBO = m/nb * svgp_lik - svgp_kl - m/nb * KL_LVM (sum over m, n_samples)
        
        
        ### ELBO = m/nb * svgp_lik - svgp_kl + m/nb prior_lik - prior_kl
        
        
        #return (m/nb * svgp_lik - svgp_kl), (m/n_b * KL)
        
        ### print normalized ELBO: ELBO / (n*m*n_samples)
        
        
        return svgp_elbo, kl

    def forward(self,
                data,
                n_mc,
                kmax=5,
                batch_idxs=None,
                ts=None,
                neuron_idxs=None):
        """
        Parameters
        ----------
        data : Tensor
            data with dimensionality (n_samples, n x m)
        n_mc : int
            number of MC samples
        kmax : int
            parameter for estimating entropy for several manifolds
            (not used for some manifolds)
        batch_idxs: Optional int list
            if None then use all data and (batch_size == m)
            otherwise, (batch_size == len(batch_idxs))

        Returns
        -------
        elbo : Tensor
            evidence lower bound of the GPLVM model averaged across MC samples and summed over n, m, n_samples (scalar)
        """

        #(n_mc, n), (n_mc)
        svgp_elbo, kl = self.elbo(data,
                                  n_mc,
                                  kmax=kmax,
                                  batch_idxs=batch_idxs,
                                  ts=ts,
                                  neuron_idxs=neuron_idxs)
        #sum over neurons, mean over  MC samples
        svgp_elbo = svgp_elbo.sum(-1).mean()
        kl = kl.mean()

        return svgp_elbo, kl  #mean across batches, sum across everything else

    def calc_LL(self, data, n_mc, kmax=5, ts=None):
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

        Returns
        -------
        LL : Tensor
            E_mc[p(Y)] (burda et al.) (scalar)
        """

        #(n_mc, n), (n_mc)
        svgp_elbo, kl = self.elbo(data,
                                  n_mc,
                                  kmax=kmax,
                                  batch_idxs=batch_idxs,
                                  ts=ts)
        print(svgp_elbo.shape, kl.shape)
        svgp_elbo = svgp_elbo.sum(-1)  #(n_mc)
        LLs = svgp_elbo - kl  # LL for each batch (n_mc)
        
        print(LLs.shape)
        LL = (torch.logsumexp(LLs, 0) - np.log(n_mc))/np.prod(data.shape)

        return LL.detach().cpu()
