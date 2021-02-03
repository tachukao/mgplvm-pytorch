from __future__ import print_function
import numpy as np
from ..utils import softplus
from . import sgp, svgp
from .. import rdist, kernels, utils
import torch
from ..manifolds.base import Manifold
from ..kernels import Kernel
from torch import nn, Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import pickle
from .. import lpriors
from typing import Optional, Tuple

jitter = utils.default_jitter


class SgpLvm(nn.Module):
    name = "SgpLvm"

    def __init__(self,
                 manif: Manifold,
                 n: int,
                 m: int,
                 n_inducing: int,
                 kernel: Kernel,
                 ref_dist,
                 sigma: Optional[Tensor] = None,
                 z: Optional[Tensor] = None,
                 lprior: Optional[lpriors.common.Lprior] = None):
        """
        __init__ method for Vanilla model

        Parameters
        ----------
        manif : Manifold
            manifold object (e.g., Euclid(1), Torus(2), so3)
        n : int
            number of neurons
        m : int
            number of conditions
        n_inducing : int
            number of inducing points
        kernel : Kernel
            kernel used for GP regression
        ref_dist : rdist
            reference distribution
        sigma : Tensor option
            optional initialization of output noise standard deviation
            output variance is then sigma^2
        lprior : lprior option
        """
        super().__init__()
        self.manif = manif
        self.d = manif.d
        self.n = n
        self.m = m
        self.n_inducing = n_inducing
        self.kernel = kernel
        self.z = self.manif.inducing_points(n, n_inducing, z=z)
        self.sgp = sgp.Sgp(self.kernel, n, m, self.z, sigma=sigma)  # sparse gp
        # reference distribution
        self.rdist = ref_dist
        self.lprior = lpriors.common.Uniform(
            manif) if lprior is None else lprior

    def forward(self, data: Tensor, n_b: int, kmax: int = 5):
        """
        Parameters
        ----------
        data : Tensor
            data with dimensionality (n_samples x n x m)
        n_b : int
            batch size
        kmax : int
            parameter for estimating entropy for several manifolds
            (not used for some manifolds)

        Returns
        -------
        sgp_elbo : Tensor
            evidence lower bound of sparse GP per batch

        kl : Tensor
            estimated KL divergence per batch between variational distribution
            and a manifold-specific prior (uniform for all manifolds except
            Euclidean, for which we have a Gaussian N(0,I) prior)

        Notes
        ----
        ELBO of the model per batch is [ sgp_elbo - kl ]
        """
        #_, _, n_samples = data.shape
        n_samples, n, m = data.shape

        q = self.rdist()  # return reference distribution

        # sample a batch with dims: (n_b x m x d)
        x = q.rsample(torch.Size([n_b]))
        print('xshape', x.shape)
        # compute entropy (n_b x m)
        lq = self.manif.log_q(q.log_prob, x, self.manif.d, kmax=kmax)

        # transform x to group with dims (n_b x m x d)
        gtilde = self.manif.expmap(x)
        # apply g_mu with dims: (n_b x m x d)
        g = self.manif.transform(gtilde)

        # sparse GP elbo summed over all batches
        # note that [ sgp.elbo ] recognizes inputs of dims (n_b x d x m)
        # and so we need to permute [ g ] to have the right dimensions
        sgp_elbo = self.sgp.elbo(n_samples, n_b, data, g.permute(0, 2, 1))

        # KL(Q(G) || p(G)) ~ logQ - logp(G)
        kl = lq.sum() - self.lprior(g).sum()
        return (sgp_elbo / n_b), (kl / n_b)

    def predict(self,
                data: Tensor,
                query: Tensor,
                niter: int = 100) -> Tuple[Tensor, Tensor]:
        _, _, n_samples = data.shape
        q = self.rdist()
        samples = []
        query = query.reshape(1, -1, self.manif.d2).permute(0, 2, 1)
        for _ in range(niter):
            x = q.sample(torch.Size([1]))
            # transform x to group with dims (1 x m x d)
            gtilde = self.manif.expmap(x)
            # apply g_mu with dims: (1 x m x d)
            g = self.manif.transform(gtilde)
            g = g.permute(0, 2, 1)  # (n_b x d x m)
            p_mu, p_cov = self.sgp.prediction(data, g, query)
            _, _, _, m_s = p_cov.shape
            p_cov = p_cov + (jitter * torch.eye(m_s).to(data.device))
            ps = [
                MultivariateNormal(p_mu[..., i], covariance_matrix=p_cov)
                for i in range(n_samples)
            ]
            l = torch.stack([p.sample() for p in ps], -1)
            samples.append(l)
        samples_all = torch.cat(samples)
        return torch.mean(samples_all, 0), torch.std(samples_all, 0)

    def calc_LL(self, data, n_b, kmax=5):
        '''
        calculate importance-weighted log likelihood as in Burda et al.
        '''

        _, _, n_samples = data.shape
        q = self.rdist()  # return reference distribution

        # sample a batch with dims: (n_b x m x d)
        x = q.rsample(torch.Size([n_b]))

        # compute entropy
        lq = self.manif.log_q(q.log_prob, x, self.manif.d, kmax=kmax)  # n_b x m
        lq = lq.sum(dim=1)  # (n_b,)

        # transform x to group with dims (n_b x m x d)
        gtilde = self.manif.expmap(x)
        # apply g_mu with dims: (n_b x m x d)
        g = self.manif.transform(gtilde)

        # compute prior
        lp = self.lprior(g)  # n_b x m
        lp = lp.sum(dim=1).to(lq.device)  # (n_b,)

        # sparse GP elbo summed over all conditions; (n_b,)
        sgp_elbo = self.sgp.elbo(n_samples,
                                 n_b,
                                 data,
                                 g.permute(0, 2, 1),
                                 tosum=False)

        logps = sgp_elbo + lp - lq  # (n_b,)
        LL = (torch.logsumexp(logps, 0) - np.log(n_b)) / (self.n * self.m)

        return LL

    def store_model(self, fname, extra_params={}):

        torch.save(self.state_dict(), fname + '.torch')

        params = {
            'model': self.name,
            'manif': self.manif.name,
            'kernel': self.kernel.name,
            'rdist': self.rdist.name,
            'n_z': self.z.n_z,
            'n': self.n,
            'm': self.m,
            'd': self.d
        }
        for key, item in extra_params.items():
            params[key] = item
        pickle.dump(params, open(fname + '.pickled', 'wb'))


#class SgpLvmComb(nn.Module):
#    name = "SgpLvmComb"
#
#    def __init__(self,
#                 manif,
#                 n,
#                 m,
#                 n_u,
#                 kernel,
#                 rdist,
#                 device,
#                 zs=None,
#                 sigma=None):
#        super().__init__()
#        self.n = n  # number of neurons
#        self.m = m  # number of conditions
#        self.n_u = n_u  # number of inducing points
#        self.manif = manif  # all the manifolds
#        self.nmanif = len(manif)
#
#        zs = [None for m in manif] if zs is None else zs
#        self.z = [
#            manif.inducing_points(n, n_u, z=z).to(device)
#            for (manif, z) in zip(self.manif, zs)
#        ]
#        # Product kernel
#        self.kernel = kernels.Product(kernel)
#        self.rdist = rdist  # reference distributions
#        self.sgp = sgp.SgpComb(self.kernel, n, m, self.z,
#                               sigma=sigma)  # sparse gp
#
#    def forward(self, data, n_b, kmax=5):
#        _, _, n_samples = data.shape
#        qs = [rdist() for rdist in self.rdist]
#        xs = [q.rsample(torch.Size([n_b])) for q in qs]
#        lqs = [
#            m.log_q(q.log_prob, x, m.d, kmax)
#            for (m, q, x) in zip(self.manif, qs, xs)
#        ]
#        gtildes = [m.expmap(x) for (m, x) in zip(self.manif, xs)]
#        gs = [m.transform(gtilde) for (m, gtilde) in zip(self.manif, gtildes)]
#
#        sgp_elbo = self.sgp.elbo(n_samples, n_b, data,
#                                 [g.permute(0, 2, 1) for g in gs])
#
#        lpriors = [
#            m.lprior(g).to(data.device) for (m, g) in zip(self.manif, gs)
#        ]
#
#        kl = torch.stack(lqs).sum() - torch.stack(lpriors).sum()
#
#        return (sgp_elbo / n_b), (kl / n_b)
#
#    def predict(self, data, query, niter: int = 100):
#        '''
#        data is training data
#        query is query points as a list of length n_manifolds
#        where each element is m_test x d_manifold
#        '''
#
#        _, _, n_samples = data.shape
#        qs = [rdist() for rdist in self.rdist]
#        samples = []
#        for _ in range(niter):
#            xs = [q.rsample(torch.Size([1])) for q in qs]
#            # transform x to group with dims (1 x m x d)
#            gtildes = [self.m.expmap(x) for (m, x) in zip(self.manif, xs)]
#            gs = [
#                m.transform(gtilde)
#                for (m, gtilde) in zip(self.manif, gtildes)
#            ]
#
#            gs = [g.permute(0, 2, 1) for g in gs]  # (1 x d x m)
#            query = [
#                q.reshape(1, m.d2, -1) for (q, m) in zip(query, self.manif)
#            ]
#
#            p_mu, p_cov = self.sgp.prediction(data, gs, query)
#            _, _, _, m_s = p_cov.shape
#            p_cov = p_cov + (1E-3 * torch.eye(m_s).to(data.device))
#            ps = [
#                MultivariateNormal(p_mu[..., i], covariance_matrix=p_cov)
#                for i in range(n_samples)
#            ]
#            l = torch.stack([p.sample() for p in ps], -1)
#            samples.append(l)
#        samples = torch.cat(samples)
#        return torch.mean(samples, 0), torch.std(samples, 0)
#
#    def calc_LL(self, data, n_b, kmax=5):
#        '''
#        calculate importance-weighted log likelihood as in Burda et al.
#        '''
#
#        _, _, n_samples = data.shape
#
#        qs = [rdist() for rdist in self.rdist]  # return reference distribution
#        # sample a batch with dims: (n_b x m x d)
#        xs = [q.rsample(torch.Size([n_b])) for q in qs]
#
#        # compute entropy
#        lqs = [
#            m.log_q(q.log_prob, x, m.d, kmax)
#            for (m, q, x) in zip(self.manif, qs, xs)
#        ]
#        lqs = [lq.sum(dim=1) for lq in lqs]  # [(n_b,), ...]
#        lq = torch.stack(lqs).sum(dim=0)  # (n_b,)
#
#        # project to group and transform
#        gtildes = [m.expmap(x) for (m, x) in zip(self.manif, xs)]
#        gs = [m.transform(gtilde) for (m, gtilde) in zip(self.manif, gtildes)]
#
#        # compute prior
#        lps = [m.lprior(g).to(data.device)
#               for (m, g) in zip(self.manif, gs)]  # (n_b, m)
#        lps = [lp.sum(dim=1) for lp in lps]  # [(n_b,), ...]
#        lp = torch.stack(lps).sum(dim=0)  # (n_b,)
#
#        # sparse GP elbo summed over all conditions; (n_b,)
#        sgp_elbo = self.sgp.elbo(n_samples,
#                                 n_b,
#                                 data, [g.permute(0, 2, 1) for g in gs],
#                                 tosum=False)
#
#        logps = sgp_elbo + lp - lq  # (n_b,)
#        LL = (torch.logsumexp(logps, 0) - np.log(n_b)) / (self.n * self.m)
#
#        return LL
#
#    def store_model(self, fname, extra_params={}, overwrite=True):
#
#        import os
#        try:
#            os.mkdir(fname)
#        except:
#            print(fname, 'already exists')
#            if overwrite:
#                print('overwriting')
#            else:
#                print('returning')
#                return
#
#        params = {
#            'model': self.name,
#            'manif': [m.name for m in self.manif],
#            'kernel': [k.name for k in self.kernel.kernels],
#            'rdist': [r.name for r in self.rdist],
#            'n_z': self.z[0].n_z,
#            'n': self.n,
#            'm': self.m,
#            'd': [m.d for m in self.manif]
#        }
#
#        for key, item in extra_params.items():
#            params[key] = item
#
#        # dump meta data
#        pickle.dump(params, open(fname + '/params.pickled', 'wb'))
#
#        # dump global paramters
#        torch.save(self.state_dict(), fname + '/model.torch')
#
#        # dump parameters for constituent manifolds
#        for i in range(self.nmanif):
#            torch.save(self.manif[i].state_dict(),
#                       fname + '/manif' + str(i) + '.torch')
#            torch.save(self.kernel.kernels[i].state_dict(),
#                       fname + '/kernel' + str(i) + '.torch')
#            torch.save(self.z[i].state_dict(),
#                       fname + '/inducing' + str(i) + '.torch')
#            torch.save(self.rdist[i].state_dict(),
#                       fname + '/rdist' + str(i) + '.torch')
