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
                 Y=None):
        """
        __init__ method for linear GPLVM with exact posteriors and Gaussian noise
        Parameters
        ----------
        """

        #observation model (P(Y|X))
        if Bayesian:
            obs = Bfa(n, d, Y=Y)  #Bayesian FA
        else:
            obs = Fa(n, d, Y=Y)  #non-Bayesian FA

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
                 tied_samples=True):
        """
        __init__ method for linear GPLVM with approximate posteriors and flexible noise models
        Parameters
        ----------
        """

        #observation model (P(Y|X))
        obs = Bvfa(n, d, m, n_samples, likelihood, tied_samples=tied_samples)
        super().__init__(obs, lat_dist, lprior, n, m, n_samples)
