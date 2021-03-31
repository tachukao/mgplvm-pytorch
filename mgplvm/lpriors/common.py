import abc
import torch
from torch import Tensor, nn
import torch.distributions as dists
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import transform_to, constraints
import numpy as np
from ..base import Module
from ..manifolds.base import Manifold


class Lprior(Module, metaclass=abc.ABCMeta):
    """
    Base kernel class
    """

    def __init__(self, manif: Manifold):
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
        '''
        uniform prior for closed manifolds, Gaussian prior for Euclidean space
        '''
        super().__init__(manif)

    def forward(self, g: Tensor, batch_idxs=None):
        lp = self.manif.lprior(g)  #(n_b, n_samples, m)
        return lp.to(g.device).sum(-1).sum(-1)  #(n_b)

    def prms(self):
        pass

    @property
    def msg(self):
        return ""


class Null(Lprior):
    name = "null"

    def __init__(self, manif):
        '''
        return 0; non-Bayesian point prior
        '''
        super().__init__(manif)

    def forward(self, g: Tensor, batch_idxs=None):
        '''
        g: (n_b x n_samples x mx x d)
        output: (n_b)
        '''
        return 0 * torch.ones(g.shape[0]).to(g.device)

    def prms(self):
        pass

    @property
    def msg(self):
        return ""


class Gaussian(Lprior):
    name = "gaussian"

    def __init__(self, manif, sigma: float = 1.5):
        '''
        Gaussian prior for Euclidean space and wrapped Gaussian for other manifolds
        Euclidean is fixed N(0, I) since the space can be scaled and rotated freely
        non-Euclidean manifolds are parameterized as ReLie[N(0, Sigma)]
        'sigma = value' initializes the sqrt diagonal elements of Sigma
        '''
        super().__init__(manif)

        #N(0,I) can always be recovered from a scaling/rotation of the space
        sigma, fixed_gamma = (1, True) if 'Euclid' in manif.name else (sigma,
                                                                       False)
        gamma = torch.diag_embed(torch.ones(1, manif.d) * sigma)
        gamma = transform_to(constraints.lower_cholesky).inv(gamma)

        if fixed_gamma:
            #don't update the covariance matrix
            self.gamma = gamma
        else:
            self.gamma = nn.Parameter(data=gamma, requires_grad=True)
        mu = torch.zeros(1, manif.d).to(gamma.device)
        self.dist = MultivariateNormal(mu, scale_tril=gamma)

    @property
    def prms(self):
        gamma = transform_to(MultivariateNormal.arg_constraints['scale_tril'])(
            self.gamma)
        return gamma

    def forward(self, g, batch_idxs=None, kmax=5):
        '''
        g: (n_samples, n_mc, m, d)
        output: (n_b)
        '''
        if g.device != self.gamma.device:
            #print('putting things on the same device')
            self.gamma = self.gamma.to(g.device)
            #print(self.dist.gamma.device, '\n')

        # return reference distribution
        q = self.dist
        #project onto tangent space
        x = self.manif.logmap(g)
        # compute log prior
        lq = self.manif.log_q(q.log_prob, g, self.manif.d,
                              kmax)  #(n_b, n_samples, m)
        return lq.sum(-1).sum(-1)

    @property
    def msg(self):
        sig = np.median(
            np.concatenate(
                [np.diag(sig) for sig in self.prms.data.cpu().numpy()]))
        return (' prior_sig {:.3f} |').format(sig)


class Brownian(Lprior):
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
        self.brownian_eta = nn.Parameter(data=torch.sqrt(brownian_eta),
                                         requires_grad=(not fixed_brownian_eta))
        self.brownian_c = nn.Parameter(data=brownian_c,
                                       requires_grad=(not fixed_brownian_c))

    @property
    def prms(self):
        brownian_eta = torch.square(self.brownian_eta) + 1E-16
        brownian_c = self.brownian_c
        return brownian_c, brownian_eta

    def forward(self, g, batch_idxs=None):
        brownian_c, brownian_eta = self.prms
        ginv = self.manif.inverse(g)
        dg = self.manif.gmul(ginv[..., 0:-1, :], g[..., 1:, :])
        dx = self.manif.logmap(dg)
        normal = dists.Normal(loc=brownian_c, scale=torch.sqrt(brownian_eta))
        diagn = dists.Independent(normal, 1)
        lq = self.manif.log_q(diagn.log_prob, dx, self.manif.d, kmax=self.kmax)
        #(n_b, n_samples, m) -> (n_b)
        return lq.sum(-1).sum(-1)

    @property
    def msg(self):
        brownian_c, brownian_eta = self.prms
        return (' brownian_c {:.3f} | brownian_eta {:.3f} |').format(
            brownian_c.detach().cpu().numpy().mean(),
            brownian_eta.detach().cpu().numpy().mean())


class ARP(Lprior):
    name = "ARP"

    def __init__(self,
                 p,
                 manif: Manifold,
                 kmax: int = 5,
                 ar_phi=None,
                 ar_eta=None,
                 ar_c=None,
                 learn_phi=True,
                 learn_eta=True,
                 learn_c=True,
                 diagonal=True):
        '''
        ..math::
          :nowrap:
          \\begin{eqnarray}
          x_t &= c + \\sum_{j=1}^p phi_j x_{t-1} + w_t \\\\
          w_t &= N(0, eta)
          \\end{eqnarray}
        '''
        super().__init__(manif)
        d = self.d
        self.p = p
        self.kmax = kmax
        if 'So3' in manif.name:
            diagonal = False
        self.diagonal = diagonal

        ar_phi = 0.0 * torch.ones(d, p) if ar_phi is None else ar_phi
        ar_eta = 0.05 * torch.ones(d) if ar_eta is None else ar_eta
        ar_c = torch.zeros(d) if ar_c is None else ar_c
        self.ar_phi = nn.Parameter(data=ar_phi, requires_grad=learn_phi)
        self.ar_eta = nn.Parameter(data=ar_eta, requires_grad=learn_eta)
        self.ar_c = nn.Parameter(data=ar_c, requires_grad=learn_c)

    @property
    def prms(self):
        return self.ar_c, self.ar_phi, torch.square(self.ar_eta)

    def forward(self, g, batch_idxs=None):
        p = self.p
        ar_c, ar_phi, ar_eta = self.prms
        ginv = self.manif.inverse(g)  # n_b x n_samplex mx x d2 (on group)
        dg = self.manif.gmul(
            ginv[..., 0:-1, :],
            g[..., 1:, :])  # n_b x n_samples x (mx-1) x d2 (on group)
        dx = self.manif.logmap(dg)  # n_b x n_samplex (mx-1) x d (on algebra)
        delta = ar_phi * torch.stack(
            [dx[..., p - j - 1:-j - 1, :] for j in range(p)], dim=-1)
        dy = dx[..., p:, :] - delta.sum(
            -1)  # n_b x n_samples x (mx-1-p) x d (on alegbra)

        scale = torch.sqrt(ar_eta)

        if self.diagonal:  #diagonal covariance
            lq = torch.stack([
                self.manif.log_q(dists.Normal(ar_c[j], scale[j]).log_prob,
                                 dy[..., j, None],
                                 1,
                                 kmax=self.kmax).sum(-1) for j in range(self.d)
            ])  #(d x n_b x n_samples x m-p-1)
            lq = lq.sum(0)  #(n_b x n_samples x m-p-1)
        else:  #not diagonal (e.g. SO(3))
            normal = dists.Normal(loc=ar_c, scale=scale)
            diagn = dists.Independent(normal, 1)
            lq = self.manif.log_q(diagn.log_prob,
                                  dy,
                                  self.manif.d,
                                  kmax=self.kmax)
            # (n_b x n_samplesx m-p-1)

        lq = lq.sum(-1).sum(-1)

        #in the future, we may want an explicit prior over the p initial points
        return lq

    @property
    def msg(self):
        ar_c, ar_phi, ar_eta = self.prms
        lp_msg = (' ar_c {:.3f} | ar_phi_avg {:.3f} | ar_eta {:.3f} |').format(
            ar_c.detach().cpu().mean(),
            ar_phi.detach().cpu().mean(),
            ar_eta.detach().cpu().sqrt().mean())
        return lp_msg
