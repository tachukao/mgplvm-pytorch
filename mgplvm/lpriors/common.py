import abc
import torch
from torch import Tensor, nn
import torch.distributions as dists
import numpy as np
from ..base import Module
from ..manifolds.base import Manifold
from ..rdist import MVN


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

    def forward(self, g: Tensor, ts=None):
        lp = self.manif.lprior(g)  #(n_b, m)
        return lp.to(g.device).sum(-1)  #(n_b)

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

    def forward(self, g: Tensor, ts=None):
        '''
        g: (n_b x mx x d)
        output: (n_b)
        '''
        return 0 * torch.ones(g.shape[0])

    def prms(self):
        pass

    @property
    def msg(self):
        return ""


class Gaussian(Lprior):
    name = "gaussian"

    def __init__(self, manif, sigma=1.5):
        '''
        Gaussian prior for Euclidean space and wrapped Gaussian for other manifolds
        Euclidean is fixed N(0, I) since the space can be scaled and rotated freely
        non-Euclidean manifolds are parameterized as ReLie[N(0, Sigma)]
        'sigma = value' initializes the sqrt diagonal elements of Sigma
        '''
        super().__init__(manif)

        if 'Euclid' in manif.name:
            #N(0,I) can always be recovered from a scaling/rotation of the space
            dist = MVN(1, manif.d, sigma=1, fixed_gamma=True)
        else:
            #parameterize the covariance matrix
            dist = MVN(1, manif.d, sigma=sigma, fixed_gamma=False)

        self.dist = dist

    @property
    def prms(self):
        return self.dist.prms

    def forward(self, g, ts=None, kmax=5):
        '''
        g: (n_b x mx x d)
        output: (n_b)
        '''
        if g.device != self.dist.gamma.device:
            #print('putting things on the same device')
            self.dist.gamma = self.dist.gamma.to(g.device)
            #print(self.dist.gamma.device, '\n')

        # return reference distribution
        q = self.dist()
        #project onto tangent space
        x = self.manif.logmap(g)
        # compute log prior
        lq = self.manif.log_q(q.log_prob, g, self.manif.d, kmax)  #(n_b, m)
        return lq.sum(-1)

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
        self.brownian_eta = nn.Parameter(
            data=torch.sqrt(brownian_eta),
            requires_grad=(not fixed_brownian_eta))
        self.brownian_c = nn.Parameter(data=brownian_c,
                                       requires_grad=(not fixed_brownian_c))

    @property
    def prms(self):
        brownian_eta = torch.square(self.brownian_eta) + 1E-16
        brownian_c = self.brownian_c
        return brownian_c, brownian_eta

    def forward(self, g, ts=None):
        brownian_c, brownian_eta = self.prms
        ginv = self.manif.inverse(g)
        dg = self.manif.gmul(ginv[..., 0:-1, :], g[..., 1:, :])
        dx = self.manif.logmap(dg)
        normal = dists.Normal(loc=brownian_c, scale=torch.sqrt(brownian_eta))
        diagn = dists.Independent(normal, 1)
        lq = self.manif.log_q(diagn.log_prob, dx, self.manif.d, kmax=self.kmax)
        #(n_b, m) -> (n_b)
        return lq.sum(-1)

    @property
    def msg(self):
        brownian_c, brownian_eta = self.prms
        return (' brownian_c {:.3f} | brownian_eta {:.3f} |').format(
            brownian_c.detach().cpu().numpy().mean(),
            brownian_eta.detach().cpu().numpy().mean())


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
        ar1_eta = torch.ones(d) if ar1_eta is None else torch.sqrt(ar1_eta)
        ar1_c = torch.zeros(d) if ar1_c is None else ar1_c
        self.ar1_phi = nn.Parameter(data=ar1_phi, requires_grad=True)
        self.ar1_eta = nn.Parameter(data=ar1_eta, requires_grad=True)
        self.ar1_c = nn.Parameter(data=ar1_c, requires_grad=True)

    @property
    def prms(self):
        return self.ar1_c, self.ar1_phi, torch.square(self.ar1_eta)

    def forward(self, g, ts=None):
        ar1_c, ar1_phi, ar1_eta = self.prms
        ginv = self.manif.inverse(g)
        dg = self.manif.gmul(ginv[..., 0:-1, :], g[..., 1:, :])
        dx = self.manif.logmap(dg)
        dy = dx[..., 1:, :] - ((ar1_phi * dx[..., 0:-1, :]))
        # dy = torch.cat((dx[..., 0:1, :], dy), -2)
        normal = dists.Normal(loc=ar1_c, scale=torch.sqrt(ar1_eta))
        diagn = dists.Independent(normal, 1)
        lq = self.manif.log_q(diagn.log_prob, dy, self.manif.d, kmax=self.kmax)
        #(n_b, m) -> (n_b)
        return lq.sum(-1)

    @property
    def msg(self):
        ar1_c, ar1_phi, ar1_eta = self.prms
        return (' ar1_c {:.3f} | ar1_phi {:.3f} | ar1_eta {:.3f} |').format(
            ar1_c.item(), ar1_phi.item(), ar1_eta.item())


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
                 learn_c=True):
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

        ar_phi = 0.0 * torch.ones(d, p) if ar_phi is None else ar_phi
        ar_eta = 0.05 * torch.ones(d) if ar_eta is None else ar_eta
        ar_c = torch.zeros(d) if ar_c is None else ar_c
        self.ar_phi = nn.Parameter(data=ar_phi, requires_grad=learn_phi)
        self.ar_eta = nn.Parameter(data=ar_eta, requires_grad=learn_eta)
        self.ar_c = nn.Parameter(data=ar_c, requires_grad=learn_c)

    @property
    def prms(self):
        return self.ar_c, self.ar_phi, torch.square(self.ar_eta)

    def forward(self, g, ts=None):
        p = self.p
        ar_c, ar_phi, ar_eta = self.prms
        ginv = self.manif.inverse(g)  # n_b x mx x d2 (on group)
        dg = self.manif.gmul(ginv[..., 0:-1, :],
                             g[..., 1:, :])  # n_b x (mx-1) x d2 (on group)
        dx = self.manif.logmap(dg)  # n_b x (mx-1) x d2 (on algebra)
        delta = ar_phi * torch.stack(
            [dx[..., p - j - 1:-j - 1, :] for j in range(p)], dim=-1)
        dy = dx[..., p:, :] - delta.sum(-1)  # n_b x (mx-1-p) x d2 (on group)

        scale = torch.sqrt(ar_eta)
        lq = torch.stack([
            self.manif.log_q(dists.Normal(ar_c[j], scale[j]).log_prob,
                             dy[..., j, None],
                             1,
                             kmax=self.kmax).sum(-1) for j in range(self.d)
        ]) #(1 x n_b x m-p-1)
        
        #print(lq.shape)
        lq = lq.sum(0).sum(-1)

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


class ARP_G(Lprior):
    name = "ARP"

    def __init__(self,
                 p,
                 manif: Manifold,
                 kmax: int = 5,
                 ar_phi=None,
                 ar_eta=None,
                 ar_c=None,
                 learn_eta=True,
                 learn_phi=True,
                 learn_c=True):
        '''
        .. math::

            \\tilde{g}_t &= g_c \\prod_{j=p}^1 \\exp(a_j \\log(g_t-j*g_{t-j-1})) g_{t-1}
            \\text{Log}[ \\tilde{g}_t g_{t-1}^(-1) ] \\sim N(0, \\eta)
        '''
        super().__init__(manif)
        d = self.d
        self.p = p
        self.kmax = kmax

        #ar_phi = 1e-3 * torch.randn(d, p-1) if ar_phi is None else ar_phi
        ar_phi = 1e-3 * torch.randn(1, p - 1) if ar_phi is None else ar_phi
        #ar_eta = 0.5 * torch.ones(d) if ar_eta is None else ar_eta
        ar_eta = torch.tensor(0.5) if ar_eta is None else ar_eta
        ar_c = 1e-3 * torch.randn(d) if ar_c is None else ar_c

        ### AR parameters
        self.ar_phi = nn.Parameter(data=ar_phi, requires_grad=learn_phi)

        #w_t \sim Exp[ N(0, eta) ]
        self.ar_eta = nn.Parameter(data=ar_eta, requires_grad=learn_eta)

        #parameterize as g_c = exp(x_c) for now
        self.ar_c = nn.Parameter(data=ar_c, requires_grad=learn_c)

    @property
    def prms(self):
        return self.ar_c, self.ar_phi, torch.square(self.ar_eta)

    def forward(self, g: Tensor, ts=None):
        '''
        g is (n_b x mx x d2)
        '''
        p = self.p
        _, mx, _ = g.shape

        ar_c, ar_phi, ar_eta = self.prms
        ginv = self.manif.inverse(g)  # n_b x mx x d2 (on group)

        if p > 1:  #if p > 1 we need to consider past displacements
            dg = self.manif.gmul(ginv[..., 0:-1, :],
                                 g[..., 1:, :])  #n_b x (mx-1) x d (on group)

            #f_i(dg) = 'a_i dg' = Exp[ a_i*Log[dg] ]
            dx = self.manif.logmap(dg)  #n_b x (mx-1) x d (on algebra)
            delta = ar_phi * torch.stack(
                [dx[..., (p - j - 2):(mx - j - 2), :] for j in range(p - 1)],
                dim=-1)  # n_b x (mx-p) x d x (p-1) (on algebra)
            delta = delta.permute(0, 1, 3,
                                  2)  # n_b x (mx-p) x (p-1) x d (on algebra)
            dg_phi = self.manif.expmap(
                delta)  #n_b x (mx-p) x (p-1) x d2 (on group)
            #print(delta.shape, dg_phi.shape)

        #consider g_{t-1} for t = (p+1:T)
        g_tm1 = g[:, (p - 1):-1, :]  # n_b x (mx-p) x d2 (on group)

        #sequentially apply group elements
        g_pred = g_tm1  #n_b x (mx-p) x d2 (on group)

        for i in range(1, p):
            #sequential multiplication ... g_t-1 * dg_t-1 * ... * dg_t-p+1
            g_pred = self.manif.gmul(
                g_pred, dg_phi[:, :, -i, :])  #n_b x (mx-p) x d2 (on group)

        #g_t-1 * dg_t-1 * ... * dg_t-p+1 * g_c
        g_c = self.manif.expmap(ar_c.reshape(1, 1, -1))  #1 x 1 x d2 (on group)
        g_pred = self.manif.gmul(
            g_pred, g_c)  #n_b x (mx-p) x d2 (on group) -- add constant element

        #g_t g_pred \approx I
        #only consider timepoints we have sufficient data for
        g_true = g[:, p:, :]  # n_b x (mx-p) x d2 (on group)
        g_pred_inv = self.manif.inverse(g_pred)  #inverse prediction
        #discrepancy between true and prediction
        g_err = self.manif.gmul(g_true,
                                g_pred_inv)  # n_b x (mx-p) x d2 (on group)

        #compute an x_err s.t. Exp(x_err) = g_err
        x_err = self.manif.logmap(g_err)  # n_b x (mx-p) x d (on group)

        #likelihood of errors is given by a diagonal Gaussian projected onto the manifold
        scale = torch.ones(self.manif.d) * torch.sqrt(ar_eta)

        lq = torch.stack([
            self.manif.log_q(dists.Normal(ar_c[j], scale[j]).log_prob,
                             x_err[..., j, None],
                             1,
                             kmax=self.kmax).sum(-1) for j in range(self.d)
        ]).sum(0)

        #return sum over m -- probability of each trajectory
        return lq

    def generate_trajectory(self, m, g0s, noise=True):
        '''
        generate trajectory of length m drawn from the model.
        must provide a 'seed' of initial states
        g0 of dim p x d2
        '''

        ar_c, ar_phi, ar_eta = self.prms
        p = self.p

        gs = torch.zeros(m + p, g0s.shape[1])
        gs[:p, :] = g0s

        noise_dist = dists.Normal(loc=0,
                                  scale=torch.ones(self.manif.d) *
                                  torch.sqrt(ar_eta))
        noises = noise_dist.sample((m + p, ))
        if not noise:
            noises *= 0

        for t in range(p, m + p):

            gprevs = gs[t - p:t, :]  #AR(p) -- need p most recent states
            if p > 1:
                ginv = self.manif.inverse(gprevs)  #p x d2
                dg = self.manif.gmul(ginv[0:-1, :],
                                     gprevs[1:, :])  #(p-1) x d2s
                dx = self.manif.logmap(dg)  #(p-1) x d (on algebra)
                #print(ar_phi.shape, dx.shape)
                delta = ar_phi.T * dx  #(p-1) x d (on algebra)
                dg_phi = self.manif.expmap(delta)  #(p-1) x d2 (on group)

            #print('dg_phi', dg_phi.shape)
            g_pred = gprevs[-1, :]  # d2 (on group)
            #print('g_pred', g_pred.shape)
            for i in range(1, p):
                #sequential multiplication ... g_t-1 * dg_t-1 * ... * dg_t-p+1
                g_pred = self.manif.gmul(g_pred, dg_phi[-i, :])  #d2 (on group)

            #g_t-1 * dg_t-1 * ... * dg_t-p+1 * g_c
            g_c = self.manif.expmap(ar_c)  # d2 (on group)
            g_pred = self.manif.gmul(
                g_pred, g_c)  # d2 (on group) -- add constant element
            #print(g_c.shape, g_pred.shape)

            x_pred = self.manif.logmap(g_pred)  # d (on algebra)
            x_new = x_pred + noises[t, :]  # d (on algebra)
            #print(x_pred.shape, x_new.shape)
            g_new = self.manif.expmap(x_new)  #new state
            #print(g_new.shape, gs.shape)
            gs[t, :] = g_new

        return gs

    @property
    def msg(self):
        ar_c, ar_phi, ar_eta = self.prms
        lp_msg = (' ar_c {:.3f} | ar_phi_avg {:.3f} | ar_eta {:.3f} |').format(
            ar_c.item(),
            torch.mean(ar_phi).item(),
            ar_eta.sqrt().item())
        return lp_msg
