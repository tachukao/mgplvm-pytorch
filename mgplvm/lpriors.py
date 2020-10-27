import abc
import torch
import torch.nn as nn
import torch.distributions as dists
from .base import Module

class AR1(Module):
    name = "AR1"

    def __init__(self, manif, kmax=5, ar1_phi=None, ar1_eta=None, ar1_c=None):
        '''
        x_t = c + phi x_{t-1} + w_t
        w_t = N(0, eta)
        '''
        super().__init__()
        d = manif.d
        self.d = d
        self.manif = manif
        self.kmax = kmax

        ar1_phi = 0.5 * torch.ones(d) if ar1_phi is None else ar1_phi
        ar1_eta = torch.ones(d) if ar1_eta is None else ar1_eta
        ar1_c = torch.ones(d) if ar1_c is None else ar1_c
        self.log_ar1_tau = nn.Parameter(data= - torch.log(- torch.log(ar1_phi)), requires_grad=True)
        self.log_ar1_eta = nn.Parameter(data = torch.log(ar1_eta), requires_grad=True)
        self.ar1_c = nn.Parameter(data=ar1_c, requires_grad=True)

    @property
    def prms(self):
        ar1_phi = torch.exp(- torch.exp(- self.log_ar1_tau))
        ar1_eta = torch.exp(self.log_ar1_eta) + 1E-16
        ar1_c = self.ar1_c
        return ar1_c, ar1_phi, ar1_eta

    def forward(self, g):
        ar1_c, ar1_phi, ar1_eta = self.prms
        ginv = self.manif.inverse(g)
        dg = self.manif.gmul(ginv[...,0:-1,:],g[...,1:,:])
        dx = self.manif.logmap(dg)
        dy = dx[...,1:,:] - ((ar1_phi * dx[...,0:-1,:]) )
        normal = dists.Normal(loc=ar1_c, scale = torch.sqrt(ar1_eta))
        diagn = dists.Independent(normal, 1)
        return self.manif.log_q(diagn.log_prob, dy, self.manif.d, kmax=self.kmax)