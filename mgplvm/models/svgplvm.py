from __future__ import print_function
import numpy as np
from mgplvm.utils import softplus
from . import sgp, svgp
from .. import rdist, kernels, utils
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import pickle
import mgplvm.lpriors as lpriors

class SvgpLvm(nn.Module):
    name = "Svgplvm"

    def __init__(self,
                 manif,
                 n,
                 m,
                 n_z,
                 kernel,
                 likelihood,
                 ref_dist,
                 lprior=None,
                 whiten=True):
        """
        __init__ method for Vanilla model
        Parameters
        ----------
        manif : Manifold
            manifold object (e.g., Euclid(1), Torus(2), so3)
        n : int
            number of neurons
        m : int
            number of conditions
        n_z : int
            number of inducing points
        kernel : Kernel
            kernel used for GP regression
        likelihood : Likelihood
            likelihood p(y|f)
        ref_dist : rdist
            reference distribution
        """
        super().__init__()
        self.manif = manif
        self.d = manif.d
        self.n = n
        self.m = m
        self.n_z = n_z
        self.kernel = kernel
        self.z = self.manif.inducing_points(n, n_z)
        self.likelihood = likelihood
        self.whiten = whiten
        self.svgp = svgp.Svgp(self.kernel,
                              n,
                              m,
                              self.z,
                              likelihood,
                              whiten=whiten)
        # reference distribution
        self.rdist = ref_dist

        #uniform is actuall gaussian for Euclidean space
        self.lprior = lpriors.Uniform(manif) if lprior is None else lprior

    def forward(self, data, n_mc, kmax=5, batch_idxs=None):
        """
        Parameters
        ----------
        data : Tensor
            data with dimensionality (n x m x n_samples)
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
            evidence lower bound of sparse GP per batch
        kl : Tensor
            estimated KL divergence per batch between variational distribution 
            and a manifold-specific prior (uniform for all manifolds except 
            Euclidean, for which we have a Gaussian N(0,I) prior)
        Notes
        ----
        ELBO of the model per batch is [ svgp_elbo - kl ]
        """

        data = data if batch_idxs is None else data[:, batch_idxs, :]
        _, _, n_samples = data.shape
        q = self.rdist(batch_idxs)  # return reference distribution

        # sample a batch with dims: (n_mc x batch_size x d)
        x = q.rsample(torch.Size([n_mc]))
        # compute entropy (summed across batches)
        lq = self.manif.log_q(q.log_prob, x, self.manif.d, kmax)

        # transform x to group with dims (n_mc x m x d)
        gtilde = self.manif.expmap(x)

        # apply g_mu with dims: (n_mc x m x d)
        g = self.manif.transform(gtilde, batch_idxs=batch_idxs)

        # sparse GP elbo summed over all batches
        # note that [ svgp.elbo ] recognizes inputs of dims (n_mc x d x m)
        # and so we need to permute [ g ] to have the right dimensions
        svgp_lik, svgp_kl = self.svgp.elbo(n_samples, n_mc, data,
                                           g.permute(0, 2, 1))
        # KL(Q(G) || p(G)) ~ logQ - logp(G)
        kl = lq.sum() - self.lprior(g).sum()
        return svgp_lik / n_mc, svgp_kl / n_mc, (kl / n_mc)
