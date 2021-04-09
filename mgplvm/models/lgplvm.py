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
from .gp_base import GpBase

from .bfa import Fa, Bfa, Bvfa
from .gplvm import Gplvm


class Lgplvm(Gplvm):
    name = "Lgplvm"

    def __init__(self,
                 n: int,
                 m: int,
                 d: int,
                 n_samples: int,
                 lat_dist: Rdist,
                 lprior: Lprior,
                 Bayesian=True,
                 Y=None,
                 learn_neuron_scale=False,
                 ard=False,
                 learn_scale=None):
        """
        __init__ method for linear GPLVM with exact posteriors and Gaussian noise
        Parameters
        ----------
        """

        #observation model (P(Y|X))
        obs = Bfa(n,
                  d,
                  Y=Y,
                  learn_neuron_scale=learn_neuron_scale,
                  ard=ard,
                  learn_scale=learn_scale) if Bayesian else Fa(n, d, Y=Y)

        super().__init__(obs, lat_dist, lprior, n, m, n_samples)


class Lvgplvm(Gplvm):
    name = "Lvgplvm"

    def __init__(self,
                 n: int,
                 m: int,
                 d: int,
                 n_samples: int,
                 lat_dist: Rdist,
                 lprior: Lprior,
                 likelihood: Likelihood,
                 tied_samples=True,
                 learn_neuron_scale=False,
                 ard=False,
                 learn_scale=None,
                 Y=None,
                rel_scale = 1):
        """
        __init__ method for linear GPLVM with approximate posteriors and flexible noise models
        Parameters
        ----------
        """

        #observation model (P(Y|X))
        obs = Bvfa(n,
                   d,
                   m,
                   n_samples,
                   likelihood,
                   tied_samples=tied_samples,
                   Y=Y,
                   learn_neuron_scale=learn_neuron_scale,
                   ard=ard,
                   learn_scale=learn_scale,
                  rel_scale = rel_scale)

        super().__init__(obs, lat_dist, lprior, n, m, n_samples)
