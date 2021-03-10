from __future__ import print_function
import numpy as np
import torch
from torch import Tensor, optim
from torch.optim.lr_scheduler import LambdaLR
from .data import DataLoader
from ..models import SvgpLvm
import itertools
from typing import Union, List, Optional


def sort_params(model, hook):
    '''apply burnin period to Sigma_Q and alpha^2
    allow for masking of certain conditions for use in crossvalidation'''

    for prm in model.lat_dist.parameters():
        prm.register_hook(hook)

    params0 = list(
        itertools.chain.from_iterable([
            model.z.parameters(),
            model.lat_dist.gmu_parameters(),
            [model.svgp.q_mu, model.svgp.q_sqrt],
        ]))

    params1 = list(
        itertools.chain.from_iterable([
            model.lat_dist.concentration_parameters(),
            model.lprior.parameters(),
            model.likelihood.parameters(),
            model.kernel.parameters()
        ]))

    params = [{'params': params0}, {'params': params1}]
    return params


def print_progress(model,
                   n,
                   m,
                   n_samples,
                   i,
                   loss_val,
                   kl_val,
                   svgp_elbo_val,
                   print_every=50,
                   Y=None,
                   batch_idxs=None,
                   sample_idxs=None):
    lat_dist = model.lat_dist
    if i % print_every == 0:
        Z = n * m * n_samples
        mu = lat_dist.lat_gmu(Y, batch_idxs=batch_idxs, sample_idxs=sample_idxs)
        gamma = lat_dist.lat_gamma(Y,
                                   batch_idxs=batch_idxs,
                                   sample_idxs=sample_idxs).diagonal(dim1=-1,
                                                                     dim2=-2)

        mu_mag = torch.sqrt(torch.mean(mu**2)).item()
        #sig = torch.median(gamma).item()
        sig = torch.median(gamma).sqrt().item()
        msg = ('\riter {:3d} | elbo {:.3f} | kl {:.3f} | loss {:.3f} ' +
               '| |mu| {:.3f} | sig {:.3f} |').format(i, svgp_elbo_val / Z,
                                                      kl_val / Z, loss_val / Z,
                                                      mu_mag, sig)
        print(msg + model.kernel.msg + model.lprior.msg +
              model.svgp.likelihood.msg,
              end="\r")


def fit(dataset: Union[Tensor, DataLoader],
        model: SvgpLvm,
        optimizer=optim.Adam,
        n_mc: int = 128,
        burnin: int = 100,
        lrate: float = 1E-3,
        max_steps: int = 1000,
        stop=None,
        print_every: int = 50,
        mask_Ts=None,
        neuron_idxs: Optional[List[int]] = None,
        prior_m=None):
    '''
    Parameters
    ----------
    dataset : Union[Tensor,DataLoader]
        data matrix of dimensions (n_samples x n x m)
    model : SvgpLvm
        model to be trained
    n_mc : int
        number of MC samples for estimating the ELBO 
    burnin : int
        number of iterations to burn in during optimization
    lrate : float
        initial learning rate passed to the optimizer
    max_steps : Optional[int], default=1000
        maximum number of training iterations
    '''

    # set learning rate schedule so sigma updates have a burn-in period
    def fburn(x):
        return 1 - np.exp(-x / (3 * burnin))

    #optionally mask some time points
    mask_Ts = mask_Ts if mask_Ts is not None else lambda x: x

    params = sort_params(model, mask_Ts)

    # instantiate optimizer
    opt = optimizer(params, lr=lrate)

    scheduler = LambdaLR(opt, lr_lambda=[lambda x: 1, fburn])

    if isinstance(dataset, torch.Tensor):
        dataloader = DataLoader(dataset)
    elif isinstance(dataset, DataLoader):
        dataloader = dataset
    else:
        raise Exception(
            "dataset passed to svgp.fit must be either a torch.Tensor or a mgplvm.optimisers.data.DataLoader"
        )

    n_samples = dataloader.n_samples
    n = dataloader.n
    m = dataloader.m

    n = n if neuron_idxs is None else len(neuron_idxs)
    for i in range(max_steps):
        loss_vals, kl_vals, svgp_vals = [], [], []
        for sample_idxs, batch_idxs, batch in dataloader:
            opt.zero_grad()
            ramp = 1 - np.exp(-i / burnin)

            svgp_elbo, kl = model(batch,
                                  n_mc,
                                  batch_idxs=batch_idxs,
                                  sample_idxs=sample_idxs,
                                  neuron_idxs=neuron_idxs,
                                  m=prior_m)

            loss = (-svgp_elbo) + (ramp * kl)  # -LL
            loss_vals.append(loss.item())
            kl_vals.append(kl.item())
            svgp_vals.append(svgp_elbo.item())
            loss.backward()
            opt.step()

        scheduler.step()
        # terminate if stop is True
        print_progress(model, n, m, n_samples, i, np.mean(loss_vals),
                       np.mean(kl_vals), np.mean(svgp_vals), print_every, batch,
                       batch_idxs, sample_idxs)
        if stop is not None:
            if stop(model, i, np.mean(loss_vals)):
                break
