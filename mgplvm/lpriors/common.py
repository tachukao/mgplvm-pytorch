import abc
import torch
import torch.nn as nn
import torch.distributions as dists
import numpy as np
from ..base import Module
from ..rdist import MVN 


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
        '''
        uniform prior for closed manifolds, Gaussian prior for Euclidean space
        '''
        super().__init__(manif)

    @property
    def prms(self):
        return None

    def forward(self, g):
        return self.manif.lprior(g)

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
        
    @property
    def prms(self):
        return None

    def forward(self, g):
        '''
        g: (n_b x mx x d)
        output: (n_b x mx)
        '''
        return 0 * torch.ones(g.shape[:2])

    @property
    def msg(self):
        return ""

class Gaussian(Lprior):
    name = "gaussian"

    def __init__(self, manif, sigma = 1.5):
        '''
        Gaussian prior for Euclidean space and wrapped Gaussian for other manifolds
        Euclidean is fixed N(0, I) since the space can be scaled and rotated freely
        non-Euclidean manifolds are parameterized as ReLie[N(0, Sigma)]
        'sigma = value' initializes the sqrt diagonal elements of Sigma
        '''
        super().__init__(manif)
        
        if 'Euclid' in manif.name:
            #N(0,I) can always be recovered from a scaling/rotation of the space
            dist = MVN(1, manif.d, sigma=1, fixed_gamma = True)
        else:
            #parameterize the covariance matrix
            dist = MVN(1, manif.d, sigma=sigma, fixed_gamma = False)
                
        self.dist = dist

    @property
    def prms(self):
        return self.dist.prms

    def forward(self, g, kmax = 5):
        '''
        g: (n_b x mx x d)
        output: (n_b x mx)
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
        lq = self.manif.log_q(q.log_prob, g, self.manif.d, kmax)
        return lq

    @property
    def msg(self):
        sig = np.median(np.concatenate([
                np.diag(sig)for sig in self.prms.data.cpu().numpy()
            ]))
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

    @property
    def msg(self):
        brownian_c, brownian_eta = self.prms
        return (' brownian_c {:.3f} | brownian_eta {:.3f} |').format(
            brownian_c.detach().cpu().numpy().mean(), brownian_eta.detach().cpu().numpy().mean())


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
        return (' ar1_c {:.3f} | ar1_phi {:.3f} | ar1_eta {:.3f} |').format(
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
        ar_eta = 0.05 * torch.ones(d) if ar_eta is None else torch.sqrt(ar_eta)
        ar_c = torch.zeros(d) if ar_c is None else ar_c
        self.ar_phi = nn.Parameter(data=ar_phi, requires_grad=True)
        self.ar_eta = nn.Parameter(data=ar_eta, requires_grad=True)
        self.ar_c = nn.Parameter(data=ar_c, requires_grad=True)

    @property
    def prms(self):
        return self.ar_c, self.ar_phi, torch.square(self.ar_eta)

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
        lp_msg = (' ar_c {:.3f} | ar_phi_avg {:.3f} | ar_eta {:.3f} |').format(
            ar_c.item(),
            torch.mean(ar_phi).item(), ar_eta.item())
        return lp_msg

    
    
class ARP_new(Lprior):
    name = "ARP"

    def __init__(self, p, manif, kmax=5, ar_phi=None, ar_eta=None, ar_c=None):
        '''
        g_t^tilde = g_c * \prod_{j=p}^1 exp(a_j*log(g_t-j*g_t-j-1)) * g_t-1
        Log[ g_t^tilde g_(t-1)^(-1) ] ~ N(0, eta)
        '''
        super().__init__(manif)
        d = self.d
        self.p = p
        self.kmax = kmax

        ar_phi = 0.0 * torch.ones(d, p) if ar_phi is None else ar_phi
        ar_eta = 0.5 * torch.ones(d) if ar_eta is None else ar_eta
        ar_c = torch.zeros(d) if ar_c is None else ar_c
        
        ### AR parameters
        self.ar_phi = nn.Parameter(data=ar_phi, requires_grad=True)
        
        #w_t \sim Exp[ N(0, eta) ]
        self.ar_eta = nn.Parameter(data=ar_eta, requires_grad=True)
        
        #parameterize as g_c = exp(x_c) for now
        self.ar_c = nn.Parameter(data=ar_c, requires_grad=True)

    @property
    def prms(self):
        return self.ar_c, self.ar_phi, torch.square(self.ar_eta)

    def forward(self, g):
        p = self.p
        ar_c, ar_phi, ar_eta = self.prms
        ginv = self.manif.inverse(g) # n_b x mx x d (on group)
        dg = self.manif.gmul(g[..., 1:, :], ginv[..., 0:-1, :]) #n_b x (mx-1) x d (on group)
        dx = self.manif.logmap(dg) #n_b x (mx-1) x d (on algebra)
        delta = ar_phi * torch.stack(
            [dx[..., (p-j-1):(-j-1), :] for j in range(p)], axis=-1) # n_b x (mx-1-p) x d x p (on algebra)
        dg_phi = self.manif.expmap(delta) #n_b x (mx-1-p) x d x p (on group)
        
        #multiply along axis -1
        dg_tot = dg_phi[:, :, :, 0] #n_b x (mx-1-p) x d (on group)
        for i in range(1, p):
            #sequential multiplication ... dg_t-3 * dg_t-2 * dg_t-1
            dg_tot = self.manif.gmul(dg_phi[:, :, :, i], dg_tot) #n_b x (mx-1-p) x d (on group)
            
        #exp(ar_c) * dg_t-p * dg_t-p+1 ... dg_t-1
        g_c = self.manif.expmap(ar_c.reshape(1, 1, -1)) #1 x 1 x d (on group)
        dg_tot = self.manif.gmul(g_c, dg_tot) #n_b x (mx-1-p) x d (on group)
        
        #g_t g_t^(-1) \approx dg_tot
        #Log[g_t g_t^(-1)] = Log[dg_tot] + wi ~ N(0, Sigma)
        Log_dg = self.manif.logmap(dg_tot) #n_b x (mx-1-p) x d (on algebra)
        
        normal = dists.Normal(loc = 0, scale=torch.sqrt(ar_eta))
        diagn = dists.Independent(normal, 1)
        
        return self.manif.log_q(diagn.log_prob,
                                dy,
                                self.manif.d,
                                kmax=self.kmax)

    @property
    def msg(self):
        ar_c, ar_phi, ar_eta = self.prms
        lp_msg = (' ar_c {:.3f} | ar_phi_avg {:.3f} | ar_eta {:.3f} |').format(
            ar_c.item(),
            torch.mean(ar_phi).item(), ar_eta.item())
        return lp_msg
