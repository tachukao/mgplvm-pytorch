import torch
from torch import Tensor
import torch.distributions
import torch.nn as nn
import abc
from .base import Module
from typing import Optional
import torch.distributions as dists
import numpy as np
from numpy.polynomial.hermite import hermgauss
import warnings
from sklearn import decomposition

log2pi: float = np.log(2 * np.pi)
n_gh_locs: int = 20  # default number of Gauss-Hermite points


def exp_link(x):
    '''exponential link function used for positive observations'''
    return torch.exp(x)


def id_link(x):
    '''identity link function used for neg binomial data'''
    return x


def FA_init(Y, d: Optional[int] = None):
    n_samples, n, m = Y.shape
    if d is None:
        d = int(np.round(n / 4))
    pca = decomposition.FactorAnalysis(n_components=d)
    Y = Y.transpose(0, 2, 1).reshape(n_samples * m, n)
    mudata = pca.fit_transform(Y)  #m*n_samples x d
    sigmas = 1.5 * np.sqrt(pca.noise_variance_)

    #print('sigma:', np.mean(sigmas), np.std(Y))

    return torch.tensor(sigmas, dtype=torch.get_default_dtype())


class Likelihood(Module, metaclass=abc.ABCMeta):

    def __init__(self, n: int, n_gh_locs: Optional[int] = n_gh_locs):
        super().__init__()
        self.n = n
        self.n_gh_locs = n_gh_locs

    @abc.abstractproperty
    def log_prob(self):
        pass

    @abc.abstractproperty
    def variational_expectation(self):
        pass

    @abc.abstractstaticmethod
    def sample(self, x: Tensor):
        pass

    @abc.abstractstaticmethod
    def dist(self, x: Tensor):
        pass

    @abc.abstractstaticmethod
    def dist_mean(self, x: Tensor):
        pass

    @property
    @abc.abstractmethod
    def msg(self):
        pass


class Gaussian(Likelihood):
    name = "Gaussian"

    def __init__(self,
                 n: int,
                 sigma: Optional[Tensor] = None,
                 n_gh_locs=n_gh_locs,
                 learn_sigma=True,
                 Y: Optional[np.ndarray] = None,
                 d: Optional[int] = None):
        super().__init__(n, n_gh_locs)

        if sigma is None:
            if Y is None:
                sigma = 1 * torch.ones(n,)
            else:
                sigma = FA_init(Y, d=d)
        self._sigma = nn.Parameter(data=sigma, requires_grad=learn_sigma)

    @property
    def prms(self) -> Tensor:
        variance = torch.square(self._sigma)
        return variance

    @property
    def sigma(self) -> Tensor:
        return (1e-20 + self.prms).sqrt()

    def log_prob(self, y):
        raise Exception("Gaussian likelihood not implemented")

    def dist(self, fs: Tensor):
        """
        Parameters
        ----------
        fs : Tensor
            GP mean function values (n_mc x n_samples x n x m)

        Returns
        -------
        dist : distribution
            resulting Gaussian distributions
        """
        prms = self.prms
        dist = torch.distributions.Normal(fs,
                                          torch.sqrt(prms)[None, None, :, None])
        return dist

    def sample(self, f_samps: Tensor) -> Tensor:
        """
        Parameters
        ----------
        f_samps : Tensor
            GP output samples (n_mc x n_samples x n x m)

        Returns
        -------
        y_samps : Tensor
            samples from the resulting Gaussian distributions (n_mc x n_samples x n x m)
        """
        dist = self.dist(f_samps)
        #sample from p(y|f)
        y_samps = dist.sample()
        return y_samps

    def dist_mean(self, fs: Tensor):
        """
        Parameters
        ----------
        fs : Tensor
            GP mean function values (n_mc x n_samples x n x m)

        Returns
        -------
        mean : Tensor
            means of the resulting Gaussian distributions (n_mc x n_samples x n x m)
            for a Gaussian, this is simply fs
        """
        return fs

    def variational_expectation(self, y, fmu, fvar):
        """
        Parameters
        ----------
        y : Tensor
            number of MC samples (n_samples x n x m)
        fmu : Tensor
            GP mean (n_mc x n_samples x n x m)
        fvar : Tensor
            GP diagonal variance (n_mc x n_samples x n x m)

        Returns
        -------
        Log likelihood : Tensor
            SVGP likelihood term per MC, neuron, sample (n_mc x n_samples x n)
        """
        n_mc, m = fmu.shape[0], fmu.shape[-1]
        variance = self.prms  #(n)
        #print(variance.shape)
        ve1 = -0.5 * log2pi * m  #scalar
        ve2 = -0.5 * torch.log(variance) * m  #(n)
        ve3 = -0.5 * torch.square(y - fmu) / variance[
            ..., None]  #(n_mc x n_samples x n x m )
        ve4 = -0.5 * fvar / variance[..., None]  #(n_mc x n_samples x n x m)

        #(n_mc x n_samples x n)
        return ve1 + ve2 + ve3.sum(-1) + ve4.sum(-1)

    @property
    def msg(self):
        sig = torch.mean(self.sigma).item()
        return (' lik_sig {:.3f} |').format(sig)


class Poisson(Likelihood):
    name = "Poisson"

    def __init__(
            self,
            n: int,
            inv_link=exp_link,  #torch.exp,
            binsize=1,
            c: Optional[Tensor] = None,
            d: Optional[Tensor] = None,
            fixed_c=True,
            fixed_d=False,
            n_gh_locs: Optional[int] = n_gh_locs):
        super().__init__(n, n_gh_locs)
        self.inv_link = inv_link
        self.binsize = binsize
        c = torch.ones(n,) if c is None else c
        d = torch.zeros(n,) if d is None else d
        self.c = nn.Parameter(data=c, requires_grad=not fixed_c)
        self.d = nn.Parameter(data=d, requires_grad=not fixed_d)
        self.n_gh_locs = n_gh_locs

    @property
    def prms(self):
        return self.c, self.d

    def log_prob(self, lamb, y):
        #lambd: (n_mc, n_samples x n, m, n_gh)
        #y: (n, n_samples x m)
        p = dists.Poisson(lamb)
        return p.log_prob(y[None, ..., None])

    def dist(self, fs: Tensor):
        """
        Parameters
        ----------
        fs : Tensor
            GP mean function values (n_mc x n_samples x n x m)

        Returns
        -------
        dist : distribution
            resulting Poisson distributions
        """
        c, d = self.prms
        lambd = self.binsize * self.inv_link(c[..., None] * fs + d[..., None])
        dist = torch.distributions.Poisson(lambd)
        return dist

    def sample(self, f_samps: Tensor):
        """
        Parameters
        ----------
        f_samps : Tensor
            GP output samples (n_mc x n_samples x n x m)

        Returns
        -------
        y_samps : Tensor
            samples from the resulting Poisson distributions (n_mc x n_samples x n x m)
        """
        dist = self.dist(f_samps)
        y_samps = dist.sample()
        return y_samps

    def dist_mean(self, fs: Tensor):
        """
        Parameters
        ----------
        fs : Tensor
            GP mean function values (n_mc x n_samples x n x m)

        Returns
        -------
        mean : Tensor
            means of the resulting Poisson distributions (n_mc x n_samples x n x m)
        """
        dist = self.dist(fs)
        mean = dist.mean.detach()
        return mean

    def variational_expectation(self, y, fmu, fvar):
        """
        Parameters
        ----------
        y : Tensor
            number of MC samples (n_samples x n x m)
        fmu : Tensor
            GP mean (n_mc x n_samples x n x m)
        fvar : Tensor
            GP diagonal variance (n_mc x n_samples x n x m)

        Returns
        -------
        Log likelihood : Tensor
            SVGP likelihood term per MC, neuron, sample (n_mc x n)
        """
        c, d = self.prms
        fmu = c[..., None] * fmu + d[..., None]
        fvar = fvar * torch.square(c[..., None])
        if self.inv_link == exp_link:
            n_mc = fmu.shape[0]
            v1 = (y * fmu) - (self.binsize * torch.exp(fmu + 0.5 * fvar))
            v2 = (y * np.log(self.binsize) - torch.lgamma(y + 1))
            #v1: (n_b x n_samples x n x m)  v2: (n_samples x n x m) (per mc sample)
            lp = v1.sum(-1) + v2.sum(-1)
            return lp

        else:
            # use Gauss-Hermite quadrature to approximate integral
            locs, ws = hermgauss(self.n_gh_locs)
            ws = torch.tensor(ws, device=fmu.device)
            locs = torch.tensor(locs, device=fvar.device)
            fvar = fvar[..., None]  #add n_gh
            fmu = fmu[..., None]  #add n_gh
            locs = self.inv_link(torch.sqrt(2. * fvar) * locs +
                                 fmu) * self.binsize  #(n_mc, n, m, n_gh)
            lp = self.log_prob(locs, y)
            return 1 / np.sqrt(np.pi) * (lp * ws).sum(-1).sum(-1)
            #return torch.sum(1 / np.sqrt(np.pi) * lp * ws)

    @property
    def msg(self):
        return " "


class ZIPoisson(Likelihood):
    """
    https://en.wikipedia.org/wiki/Zero-inflated_model
    """
    name = "Zero-inflated Poisson"

    def __init__(
            self,
            n: int,
            inv_link=exp_link,  #torch.exp,
            binsize=1,
            c: Optional[Tensor] = None,
            d: Optional[Tensor] = None,
            fixed_c=True,
            fixed_d=False,
            alpha: Optional[Tensor] = None,
            learn_alpha=True,
            n_gh_locs: Optional[int] = n_gh_locs):
        super().__init__(n, n_gh_locs)
        self.inv_link = inv_link
        self.binsize = binsize
        c = torch.ones(n,) if c is None else c
        d = torch.zeros(n,) if d is None else d
        self.c = nn.Parameter(data=c, requires_grad=not fixed_c)
        self.d = nn.Parameter(data=d, requires_grad=not fixed_d)
        self.n_gh_locs = n_gh_locs

        alpha = torch.zeros(
            n,) if alpha is None else alpha  # zero inflation probability
        self.alpha = nn.Parameter(alpha, requires_grad=learn_alpha)

    @property
    def prms(self):
        return dists.transform_to(dists.constraints.interval(0., 1.))(
            self.alpha), self.c, self.d

    def log_prob(self, lamb, y, alpha):
        #lambd: (n_mc, n_samples x n, m, n_gh)
        #y: (n, n_samples x m)
        p = dists.Poisson(lamb)
        Y = y[None, ..., None]
        zero_Y = (Y == 0)
        alpha_ = alpha[None, None, :, None, None]

        alpha_logp = torch.log(1 - alpha_) + p.log_prob(Y)  # range -infty to 0
        logp_0 = zero_Y * torch.log(alpha_ + torch.exp(alpha_logp) + 1e-12)
        logp_rest = (~zero_Y) * alpha_logp
        return logp_0 + logp_rest

    def dist(self, fs: Tensor):
        """
        Parameters
        ----------
        fs : Tensor
            GP mean function values (n_mc x n_samples x n x m)

        Returns
        -------
        dist : distribution
            resulting Poisson distributions
        """
        _, c, d = self.prms
        lambd = self.binsize * self.inv_link(c[..., None] * fs + d[..., None])
        dist = torch.distributions.Poisson(lambd)
        return dist

    def sample(self, f_samps: Tensor):
        """
        Parameters
        ----------
        f_samps : Tensor
            GP output samples (n_mc x n_samples x n x m)

        Returns
        -------
        y_samps : Tensor
            samples from the resulting Poisson distributions (n_mc x n_samples x n x m)
        """
        alpha, _, _ = self.prms
        alpha_ = alpha[None, None, :, None].expand(*f_samps.shape)
        bern = dists.Bernoulli(probs=alpha_)
        dist = self.dist(f_samps)
        y_samps = dist.sample()
        zero_inflates = 1 - bern.sample()
        return zero_inflates * y_samps

    def dist_mean(self, fs: Tensor):
        """
        Parameters
        ----------
        fs : Tensor
            GP mean function values (n_mc x n_samples x n x m)

        Returns
        -------
        mean : Tensor
            means of the resulting ZIP distributions (n_mc x n_samples x n x m)
        """
        alpha, _, _ = self.prms
        dist = (1 - alpha)[None, None, :, None] * self.dist(fs)
        mean = dist.mean.detach()
        return mean

    def variational_expectation(self, y, fmu, fvar):
        """
        Parameters
        ----------
        y : Tensor
            number of MC samples (n_samples x n x m)
        fmu : Tensor
            GP mean (n_mc x n_samples x n x m)
        fvar : Tensor
            GP diagonal variance (n_mc x n_samples x n x m)

        Returns
        -------
        Log likelihood : Tensor
            SVGP likelihood term per MC, neuron, sample (n_mc x n)
        """
        alpha, c, d = self.prms
        fmu = c[..., None] * fmu + d[..., None]
        fvar = fvar * torch.square(c[..., None])

        # use Gauss-Hermite quadrature to approximate integral
        locs, ws = hermgauss(self.n_gh_locs)
        ws = torch.tensor(ws, device=fmu.device)
        locs = torch.tensor(locs, device=fvar.device)
        fvar = fvar[..., None]  #add n_gh
        fmu = fmu[..., None]  #add n_gh
        locs = self.inv_link(torch.sqrt(2. * fvar) * locs +
                             fmu) * self.binsize  #(n_mc, n, m, n_gh)
        lp = self.log_prob(locs, y, alpha)
        return 1 / np.sqrt(np.pi) * (lp * ws).sum(-1).sum(-1)
        #return torch.sum(1 / np.sqrt(np.pi) * lp * ws)

    @property
    def msg(self):
        return " "


class NegativeBinomial(Likelihood):
    name = "Negative binomial"

    def __init__(self,
                 n: int,
                 inv_link=id_link,
                 binsize=1,
                 total_count: Optional[Tensor] = None,
                 c: Optional[Tensor] = None,
                 d: Optional[Tensor] = None,
                 fixed_total_count=False,
                 fixed_c=True,
                 fixed_d=False,
                 n_gh_locs: Optional[int] = n_gh_locs,
                 Y: Optional[np.ndarray] = None):
        super().__init__(n, n_gh_locs)
        self.inv_link = inv_link
        self.binsize = binsize

        ###initialize total_counts
        if total_count is None:
            if Y is None:
                total_count = 4 * torch.ones(n,)
            else:  #assume p = 0.5; mean = total_count
                total_count = torch.tensor(np.mean(Y, axis=(0, -1)))

        total_count = dists.transform_to(
            dists.constraints.greater_than_eq(0)).inv(total_count)
        assert (total_count is not None)
        self.total_count = nn.Parameter(data=total_count,
                                        requires_grad=not fixed_total_count)

        c = torch.ones(n,) if c is None else c
        d = torch.zeros(n,) if d is None else d
        self.c = nn.Parameter(data=c, requires_grad=not fixed_c)
        self.d = nn.Parameter(data=d, requires_grad=not fixed_d)

    @property
    def prms(self):
        total_count = dists.transform_to(dists.constraints.greater_than_eq(0))(
            self.total_count)
        return total_count, self.c, self.d

    def dist(self, fs: Tensor):
        """
        Parameters
        ----------
        fs : Tensor
            GP mean function values (n_mc x n_samples x n x m)

        Returns
        -------
        dist : distribution
            resulting negative binomial distributions
        """
        total_count, c, d = self.prms
        rate = c[..., None] * fs + d[..., None]  #shift+scale
        rate = self.inv_link(rate) * self.binsize
        dist = dists.NegativeBinomial(total_count[None, None, ..., None],
                                      logits=rate)  #neg binom
        return dist

    def sample(self, f_samps: Tensor):
        """
        Parameters
        ----------
        f_samps : Tensor
            GP output samples (n_mc x n_samples x n x m)

        Returns
        -------
        y_samps : Tensor
            samples from the resulting negative binomial distributions (n_mc x n_samples x n x m)
        """
        dist = self.dist(f_samps)
        y_samps = dist.sample()  #sample observations
        return y_samps

    def dist_mean(self, fs: Tensor):
        """
        Parameters
        ----------
        fs : Tensor
            GP mean function values (n_mc x n_samples x n x m)

        Returns
        -------
        mean : Tensor
            means of the resulting negative binomial distributions (n_mc x n_samples x n x m)
        """
        dist = self.dist(fs)
        mean = dist.mean.detach()
        return mean

    def log_prob(self, total_count, rate, y):
        #total count: (n) -> (n_mc, n_samples, n, m, n_gh)
        #rate: (n_mc, n_samples, n, m, n_gh)
        #y: (n_samples, n, m)
        p = dists.NegativeBinomial(total_count[None, ..., None, None],
                                   logits=rate)
        return p.log_prob(y[None, ..., None])

    def variational_expectation(self, y, fmu, fvar):
        """
        Parameters
        ----------
        y : Tensor
            number of MC samples (n_samples x n x m)
        fmu : Tensor
            GP mean (n_mc x n_samples x n x m)
        fvar : Tensor
            GP diagonal variance (n_mc x n_samples x n x m)

        Returns
        -------
        Log likelihood : Tensor
            SVGP likelihood term per MC, neuron, sample (n_mc x n_samples x n)
        """
        total_count, c, d = self.prms
        fmu = c[..., None] * fmu + d[..., None]
        fvar = fvar * torch.square(c[..., None])
        #print(fmu.shape, fvar.shape)
        # use Gauss-Hermite quadrature to approximate integral
        locs, ws = hermgauss(
            self.n_gh_locs)  #sample points and weights for quadrature
        ws = torch.tensor(ws, device=fmu.device)
        locs = torch.tensor(locs, device=fvar.device)
        fvar = fvar[..., None]  #add n_samples and locs
        fmu = fmu[..., None]  #add locs
        #print(locs.shape)
        locs = self.inv_link(torch.sqrt(2. * fvar) * locs +
                             fmu) * self.binsize  #coordinate transform
        #print(total_count.shape, locs.shape)
        lp = self.log_prob(total_count, locs,
                           y)  #(n_mc x n_samples x n x m, n_gh)

        #print(lp.shape, ws.shape, (lp * ws).shape)
        return 1 / np.sqrt(np.pi) * (lp * ws).sum(-1).sum(-1)

    @property
    def msg(self):
        total_count = torch.mean(self.prms[0]).item()
        return (' lik_count {:.3f} |').format(total_count)
