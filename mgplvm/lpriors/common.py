import abc
import torch
import torch.nn as nn
import torch.distributions as dists
from ..base import Module


class Lprior(Module, metaclass=abc.ABCMeta):
    """
    Base kernel class
    """
    def __init__(self, manif):
        super().__init__()
        self.manif = manif
        self.d = self.manif.d

    @property
    @abc.abstractmethod
    def msg(self):
        pass


class Uniform(Lprior):
    name = "uniform"

    def __init__(self, manif):
        super().__init__(manif)

    @property
    def prms(self):
        return None

    def forward(self, g):
        return self.manif.lprior(g)

    @property
    def msg(self):
        return ""


class Brownian(Lprio):
    name = "Brownian"

    def __init__(self,
                 manif,
                 kmax=5,
                 brownian_eta=None,
                 brownian_c=None,
                 fixed_brownian_eta=False,
                 fixed_brownian_c=False):
        '''
        x_t = c + w_t
        w_t = N(0, eta)
        '''
        super().__init__(manif)
        self.kmax = kmax
        d = self.d

        brownian_eta = torch.ones(d) if brownian_eta is None else brownian_eta
        brownian_c = torch.zeros(d) if brownian_c is None else brownian_c
        self.log_brownian_eta = nn.Parameter(
            data=torch.log(brownian_eta),
            requires_grad=(not fixed_brownian_eta))
        self.brownian_c = nn.Parameter(data=brownian_c,
                                       requires_grad=(not fixed_brownian_c))

    @property
    def prms(self):
        brownian_eta = torch.exp(self.log_brownian_eta) + 1E-16
        brownian_c = self.brownian_c
        return brownian_c, brownian_eta

    def forward(self, g):
        brownian_c, brownian_eta = self.prms
        ginv = self.manif.inverse(g)
        dg = self.manif.gmul(ginv[..., 0:-1, :], g[..., 1:, :])
        dx = self.manif.logmap(dg)
        normal = dists.Normal(loc=brownian_c, scale=torch.sqrt(brownian_eta))
        diagn = dists.Independent(normal, 1)
        return self.manif.log_q(diagn.log_prob,
                                dx,
                                self.manif.d,
                                kmax=self.kmax)

    def msg(self):
        brownian_c, brownian_eta = self.prms
        return ('brownian_c {:.3f} | brownian_eta {:.3f}').format(
            brownian_c.item(), brownian_eta.item())


class AR1(Lprior):
    name = "AR1"

    def __init__(self, manif, kmax=5, ar1_phi=None, ar1_eta=None, ar1_c=None):
        '''
        x_t = c + phi x_{t-1} + w_t
        w_t = N(0, eta)
        '''
        super().__init__(manif)
        d = self.d
        self.kmax = kmax

        ar1_phi = 0.0 * torch.ones(d) if ar1_phi is None else ar1_phi
        ar1_eta = torch.ones(d) if ar1_eta is None else ar1_eta
        ar1_c = torch.zeros(d) if ar1_c is None else ar1_c
        self.ar1_phi = nn.Parameter(data=ar1_phi, requires_grad=True)
        self.ar1_eta = nn.Parameter(data=ar1_eta, requires_grad=True)
        self.ar1_c = nn.Parameter(data=ar1_c, requires_grad=True)

    @property
    def prms(self):
        return self.ar1_c, self.ar1_phi, self.ar1_eta

    def forward(self, g):
        ar1_c, ar1_phi, ar1_eta = self.prms
        ginv = self.manif.inverse(g)
        dg = self.manif.gmul(ginv[..., 0:-1, :], g[..., 1:, :])
        dx = self.manif.logmap(dg)
        dy = dx[..., 1:, :] - ((ar1_phi * dx[..., 0:-1, :]))
        # dy = torch.cat((dx[..., 0:1, :], dy), -2)
        normal = dists.Normal(loc=ar1_c, scale=torch.sqrt(ar1_eta))
        diagn = dists.Independent(normal, 1)
        return self.manif.log_q(diagn.log_prob,
                                dy,
                                self.manif.d,
                                kmax=self.kmax)

    @property
    def msg(self):
        ar1_c, ar1_phi, ar1_eta = self.prms
        return ('ar1_c {:.3f} | ar1_phi {:.3f} | ar1_eta {:.3f}').format(
            ar1_c.item(), ar1_phi.item(), ar1_eta.item())


class ARP(Lprior):
    name = "ARP"

    def __init__(self, p, manif, kmax=5, ar_phi=None, ar_eta=None, ar_c=None):
        '''
        x_t = c + \sum_{j=1}^p phi_j x_{t-1} + w_t
        w_t = N(0, eta)
        '''
        super().__init__(manif)
        d = self.d
        self.p = p
        self.kmax = kmax

        ar_phi = 0.0 * torch.ones(d, p) if ar_phi is None else ar_phi
        ar_eta = 0.05 * torch.ones(d) if ar_eta is None else ar_eta
        ar_c = torch.zeros(d) if ar_c is None else ar_c
        self.ar_phi = nn.Parameter(data=ar_phi, requires_grad=True)
        self.ar_eta = nn.Parameter(data=ar_eta, requires_grad=True)
        self.ar_c = nn.Parameter(data=ar_c, requires_grad=True)

    @property
    def prms(self):
        return self.ar_c, self.ar_phi, self.ar_eta

    def forward(self, g):
        p = self.p
        ar_c, ar_phi, ar_eta = self.prms
        ginv = self.manif.inverse(g)
        dg = self.manif.gmul(ginv[..., 0:-1, :], g[..., 1:, :])
        dx = self.manif.logmap(dg)
        delta = ar_phi * torch.stack(
            [dx[..., p - j - 1:-j - 1, :] for j in range(p)], axis=-1)
        dy = dx[..., p:, :] - delta.sum(-1)
        normal = dists.Normal(loc=ar_c, scale=torch.sqrt(ar_eta))
        diagn = dists.Independent(normal, 1)
        return self.manif.log_q(diagn.log_prob,
                                dy,
                                self.manif.d,
                                kmax=self.kmax)

    @property
    def msg(self):
        ar_c, ar_phi, ar_eta = self.prms
        lp_msg = ('ar_c {:.3f} | ar_phi_avg {:.3f} | ar_eta {:.3f}').format(
            ar_c.item(),
            torch.mean(ar_phi).item(), ar_eta.item())
        return lp_msg
