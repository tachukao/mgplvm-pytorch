from __future__ import print_function
import numpy as np
from ..utils import softplus
from . import svgp
from .. import lat_dist, kernels, utils
import torch
from torch import nn, Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import pickle
from .. import priors
from ..inducing_variables import InducingPoints
from ..kernels import Kernel
from ..likelihoods import Likelihood
from ..priors.common import Prior
from ..lat_dist import LatentDistribution

from .gplvm import Gplvm


class SvgpLvm(Gplvm):
    name = "Svgplvm"

    def __init__(self,
                 n: int,
                 m: int,
                 n_samples: int,
                 z: InducingPoints,
                 kernel: Kernel,
                 likelihood: Likelihood,
                 lat_dist: LatentDistribution,
                 prior: Prior,
                 whiten: bool = True,
                 tied_samples=True):
        """
        __init__ method for GPLVM model with svgp observation model
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
        lat_dist : LatentDistribution
            latent distribution
        prior: Prior
            prior over the latents
        whiten: bool
            parameter passed to Svgp
        """

        #p(Y|X)
        obs = svgp.Svgp(kernel,
                        n,
                        m,
                        n_samples,
                        z,
                        likelihood,
                        whiten=whiten,
                        tied_samples=tied_samples)

        super().__init__(obs, lat_dist, prior, n, m, n_samples)
