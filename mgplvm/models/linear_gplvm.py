from __future__ import print_function
from typing import Union
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
from .gp_base import GPBase
from .factor_analysis import FA, BFA, BVFA, VFA
from .gplvm import GPLVM


class LGPLVM(GPLVM):
    name = "Lgplvm"

    def __init__(self,
                 n: int,
                 m: int,
                 d: int,
                 n_samples: int,
                 lat_dist: LatentDistribution,
                 prior: Prior,
                 Bayesian=True,
                 Y=None,
                 learn_neuron_scale=False,
                 ard=False,
                 learn_scale=None,
                 sigma=None,
                 C=None):
        """
        __init__ method for linear GPLVM with exact posteriors and Gaussian noise
        Parameters
        ----------
        """

        #observation model (P(Y|X))
        obs = BFA(n,
                  d,
                  Y=Y,
                  learn_neuron_scale=learn_neuron_scale,
                  ard=ard,
                  learn_scale=learn_scale) if Bayesian else FA(
                      n, d, Y=Y, sigma=sigma, C=C)

        super().__init__(obs, lat_dist, prior, n, m, n_samples)


LVGPLVM_OBSERVATION = Union[BVFA, VFA]


class LVGPLVM(GPLVM):
    name = "Lvgplvm"

    def __init__(self,
                 n: int,
                 m: int,
                 d: int,
                 n_samples: int,
                 lat_dist: LatentDistribution,
                 prior: Prior,
                 likelihood: Likelihood,
                 tied_samples=True,
                 learn_neuron_scale=False,
                 ard=False,
                 learn_scale=None,
                 Y=None,
                 rel_scale=1,
                 Bayesian=True,
                 C=None,
                 q_mu=None,
                 q_sqrt=None,
                 scale=None,
                 dim_scale=None,
                 neuron_scale=None):
        """
        __init__ method for linear GPLVM with approximate posteriors and flexible noise models
        Parameters
        ----------
        """

        #observation model (P(Y|X))

        if Bayesian:
            obs: LVGPLVM_OBSERVATION = BVFA(
                n,
                d,
                m,
                n_samples,
                likelihood,
                tied_samples=tied_samples,
                Y=Y,
                learn_neuron_scale=learn_neuron_scale,
                ard=ard,
                learn_scale=learn_scale,
                rel_scale=rel_scale,
                q_mu=q_mu,
                q_sqrt=q_sqrt,
                scale=scale,
                dim_scale=dim_scale,
                neuron_scale=neuron_scale)
        else:
            obs = VFA(n,
                      d,
                      m,
                      n_samples,
                      likelihood,
                      rel_scale=rel_scale,
                      Y=Y,
                      C=C)

        super().__init__(obs, lat_dist, prior, n, m, n_samples)
