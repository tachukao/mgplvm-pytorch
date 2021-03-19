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

from .bfa import fa, Bfa, Bvfa
from .svgplvm import Svgplvm


class LgpLvm(Svgplvm):
    name = "Lgplvm"

    def __init__(self,
                 n: int,
                 m: int,
                 d: int,
                 n_samples: int,
                 lat_dist: Rdist,
                 lprior: Lprior,
                 likelihood: Likelihood = None,
                 whiten: bool = True,
                 tied_samples=True,
                svgp = None,
                stochastic = True,
                Y = None):
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
        """
        self.n = n
        self.m = m
        self.n_samples = n_samples
        
        #observation model
        if stochastic:
            self.svgp = Bvfa(n,d,m,n_samples,likelihood: Likelihood, tied_samples=tied_samples)
        elif Bayesian:
            self.svgp = Bfa(n,d,Y = Y)
        else:
            self.svgp = Fa(n,d,Y = Y)
        
        # latent distribution
        self.lat_dist = lat_dist
        self.lprior = lprior