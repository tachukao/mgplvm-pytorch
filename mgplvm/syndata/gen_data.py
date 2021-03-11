import abc
from scipy.linalg import norm
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import subprocess
import torch
from torch.distributions import Bernoulli


# %% base class
def draw_GP(n, d, n_samples, sig, ell, jitter=1e-6):
    '''draw RBF GP samples with (n*n_samples) x samples, ell = l (in units of samples)'''
    rep_ts = np.arange(n).reshape(-1, 1).repeat(n, 1)
    dts = rep_ts - rep_ts.T
    K = sig**2 * np.exp(-dts**2 / (2 * ell**2))  #nxn
    L = np.linalg.cholesky(K + jitter * np.eye(n))  #nxn

    us = np.random.normal(size=(n_samples, n, d))  #n_samples x n x d
    X = L @ us  #n_samples x n x d (numpy deals with the shapes correctly here)
    return X


class Manif(metaclass=abc.ABCMeta):

    def __init__(self, d):
        self.d = d

    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def gen(self, n, n_samples, ell, sig):
        pass

    @abc.abstractmethod
    def distance(self, x, y):
        pass


# %% Euclidean class


class Euclid(Manif):

    def __init__(self, d):
        super().__init__(d)

    @property
    def name(self):
        return "euclid(%i)" % self.d

    def gen(self, n, n_samples, ell=None, sig=10, prefs=False):
        if ell is None:
            #gs = np.random.uniform(0, 4, size=(n_samples, n, self.d))
            #gs = np.random.normal(2, 1, size=(n_samples, n, self.d))
            if prefs:
                gs = np.random.uniform(-2.5, 2.5, size=(n_samples, n, self.d))
            else:
                gs = np.random.normal(0, 1, size=(n_samples, n, self.d))
        else:
            gs = draw_GP(n, self.d, n_samples, sig, ell)
            #gs = gs - np.mean(gs, axis=0) + 2
            gs = gs - np.mean(gs, axis=0)
        return gs

    def gen_ginit(self, n, n_samples):
        gs = np.zeros((n_samples, n, self.d))
        return gs

    def noisy_conds(self, gs, variability):
        gs = gs + np.random.normal(0, np.std(gs) * variability, size=gs.shape)
        return gs

    def distance(self, x, y):
        return np.sum((x[:, :, None, ...] - y[:, None, ...])**2, axis=-1)


class Torus(Manif):

    def __init__(self, d):
        super().__init__(d)

    @property
    def name(self):
        return ("torus(d)" % self.d)

    def norm(self, gs):
        return gs % (2 * np.pi)

    def gen(self, n, n_samples, ell=None, sig=10, prefs=False):
        """if l is none, draw random samples - otherwise draw from an RBF GP with ell = l"""
        if ell is None:
            gs = np.random.uniform(0, 2 * np.pi, size=(n_samples, n, self.d))
        else:
            gs = draw_GP(n, self.d, n_samples, sig, ell)
            gs = (gs + np.ceil(10 * sig) * 2 * np.pi) % (
                2 * np.pi)  #put back on the torus

        return gs

    def gen_ginit(self, n, n_samples):
        gs = np.ones((n_samples, n, self.d)) * np.pi
        return gs

    def noisy_conds(self, gs, variability):
        gs = gs + np.random.normal(0, np.std(gs) * variability, size=gs.shape)
        return self.norm(gs)

    def distance(self, x, y):
        return np.sum(2 - 2 * np.cos(x[:, :, None, ...] - y[:, None, ...]),
                      axis=-1)


# %% Sphere class


class Sphere(Manif):

    def __init__(self, d):
        super().__init__(d)

    @property
    def name(self):
        return ("sphere(d)" % self.d)

    def norm(self, gs):
        gs = gs / norm(gs, axis=1, keepdims=True)
        return gs

    def noisy_conds(self, gs, variability):
        gs = gs + np.random.normal(0, np.std(gs) * variability, size=gs.shape)
        return self.norm(gs)

    def gen(self, n, n_samples, ell=None, sig=None, prefs=False):
        '''generate random points in spherical space according to the prior'''
        gs = np.random.normal(0, 1, size=(n_samples, n, self.d + 1))
        return self.norm(gs)

    def gen_ginit(self, n, n_samples):
        gs = np.zeros((n_samples, n, self.d + 1))
        gs[:, 0] = 1
        return gs

    def distance(self, x, y):
        4 * (1 - np.sum(x[:, :, None, ...] * y[:, None, ...], axis=-1)**2)
        return 2 * (1 - np.sum(x[:, :, None, ...] * y[:, None, ...], axis=-1))


# %% SO(3) class


class So3(Manif):

    def __init__(self):
        super().__init__(3)

    @property
    def name(self):
        return "SO(3)"

    def norm(self, gs):
        gs = gs / norm(gs, axis=1, keepdims=True)
        gs = gs * np.sign(gs[..., :1])
        return gs

    def gen(self, n, n_samples, ell=None, sig=None, prefs=False):
        '''generate random points in spherical space according to the prior'''
        gs = np.random.normal(0, 1, size=(n_samples, n, self.d + 1))
        return self.norm(gs)

    def gen_ginit(self, n, n_samples):
        gs = np.zeros((n_samples, n, self.d + 1))
        gs[:, 0] = 1
        return gs

    def noisy_conds(self, gs, variability):
        gs = gs + np.random.normal(0, np.std(gs) * variability, size=gs.shape)
        return self.norm(gs)

    def distance(self, x, y):
        ds = 4 * (1 - np.sum(x[:, :, None, ...] * y[:, None, ...], axis=-1))
        return ds


# %% Product class


class Product(Manif):
    """
    Does not support product of products at the moment
    """

    def __init__(self, manifs):
        self.ds = [m.d for m in manifs]
        self.d = sum(self.ds)
        self.manifs = manifs

    @property
    def name(self):
        names = "x".join([m.name for m in self.manifs])
        return names

    def gen(self, n, n_samples, ell=None, sig=10, prefs=False):
        gs = [
            m.gen(n, n_samples, ell=ell, sig=sig, prefs=prefs)
            for m in self.manifs
        ]
        return gs

    def gen_ginit(self, n, n_samples):
        '''generate a series of points at the origin'''
        gs = [m.gen_ginit(n, n_samples) for m in self.manifs]
        return gs

    def add_tangent_vector(self, gs, d, delta):
        gs = [
            m.add_tangent_vector(g, d, delta)
            for (m, g) in zip(self.manifs, gs)
        ]
        return gs

    def distance(self, xs, ys):
        return [m.distance(x, y) for (m, x, y) in zip(self.manifs, xs, ys)]

    def distance_scaled(self, xs, ys, ls):
        return [
            m.distance(x, y) / (l**2)
            for (m, x, y, l) in zip(self.manifs, xs, ys, ls)
        ]


# %% generator class
class Gen():

    def __init__(self,
                 manifold,
                 n,
                 m,
                 l=0.5,
                 alpha=1,
                 beta=0.3,
                 sigma=0.1,
                 variability=0.1,
                 n_samples=1):
        '''
        initialize a data-generating object
        manifold is an instance of a manifold class or a list of manifolds
        '''
        # make sure everything is in list format to allow for product kernels etc.
        if isinstance(manifold, Manif):
            manifold = [manifold]
        self.nman = len(manifold)

        self.manifold = Product(manifold)
        self.variability = variability
        self.n_samples = n_samples
        self.n, self.m = n, m
        self.gprefs = []
        self.gs = []
        self.params = {}
        for (param, value) in [('alpha', alpha), ('beta', beta),
                               ('sigma', sigma), ('l', l)]:
            self.set_param(param, value)

    def get_params(self):
        '''return parameters'''
        return self.params

    def set_param(self, param, value):
        '''set the value of a parameter'''
        if param == "l":  # need separate lengthscale per neuron per manifold
            if type(value) in [int, float, np.float64
                              ]:  # one length scale for each manifold
                value = [value for i in range(self.nman)]
            value = [
                np.ones((self.n, 1)) * val +
                np.random.normal(0, val * self.variability, size=(self.n, 1))
                for val in value
            ]
        elif param == 'beta':  # one param for each neuron
            value = np.random.uniform(0, value, size=(self.n, 1))  # uniform
        else:
            value = np.ones((self.n, 1))*value + \
                np.random.normal(0, value*self.variability, size=(self.n, 1))

        # save result
        self.params[param] = value

    def gen_gprefs(self):
        '''generate prefered directions for each neuron'''
        self.gprefs = self.manifold.gen(self.n, self.n_samples, prefs=True)
        return self.gprefs

    def gen_gconds(self, ell=None, sig=10):
        '''generate conditions for each neuron'''
        self.gs = self.manifold.gen(self.m, self.n_samples, ell=ell, sig=sig)
        return self.gs

    def noisy_conds(self):
        '''
        add noise to the conditions
        '''
        gs = [
            self.manifold.manifs[i].noisy_conds(self.gs[i], self.variability)
            for i in range(self.nman)
        ]
        return gs

    def gen_data(self,
                 gs_in=None,
                 gprefs_in=None,
                 mode='Gaussian',
                 overwrite=True,
                 sigma=None,
                 ell=None,
                 sig=10,
                 rate=10):
        """
        tbin is time of each time step (by default each time step is 1 ms)
        gs_in is optional input latent signal, otherwise random points on manifold
        generate Gaussian noise neural activities
        generate IPP spiking from Gaussian bump rate model
        rate is the mean peak firing rate across neurons
        """
        # generate tuning curves
        if not overwrite:
            gprefs_backup, gs_backup = self.gprefs, self.gs

        ### Generating G according to p(G)
        if gs_in is None:
            if len(self.gs) == 0:
                self.gen_gconds(ell=ell, sig=sig)
        else:
            self.gs = gs_in

        if gprefs_in is None:
            if len(self.gprefs) == 0:
                self.gen_gprefs()
        else:
            self.gprefs = gprefs_in

        ### Generating according to p(F | G)
        ds_sqr = np.array(
            self.manifold.distance_scaled(
                self.gprefs, self.gs,
                self.params['l']))  # nman x n_samples x n x m
        Ks = np.exp(-0.5 * ds_sqr)  # nman x n_samples x n x m
        K = np.prod(Ks, axis=0)  # n_samples x n x m
        n_samples, n, m = K.shape
        fs = self.params['alpha'] * K + self.params['beta']
        self.fs = fs  #store denoised activities

        ### Generating according to p(Y | F)
        if mode == 'Gaussian':
            # add Gaussian noise
            sigma = self.params['sigma'] if sigma is None else sigma
            noise = np.random.normal(
                0,
                np.repeat(np.repeat(sigma[None, ...], m, axis=-1),
                          n_samples,
                          axis=0))
            self.Y = fs + noise
        elif mode == 'Poisson':  # Poisson spiking
            max_activity = np.mean(np.amax(fs, axis=1))
            #draw poisson samples
            self.Y = np.random.poisson(fs * rate / max_activity)
        else:
            raise NotImplementedError('Synthetic data type not supported.')

        if not overwrite:  # reset
            self.gprefs, self.gs = gprefs_backup, gs_backup

        return self.Y

    def get_data(self):
        return self.Y
