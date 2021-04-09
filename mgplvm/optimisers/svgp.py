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

    if not ('GP' in model.lat_dist.name):
        for prm in model.lat_dist.parameters():
            prm.register_hook(hook)

    params0 = list(
        itertools.chain.from_iterable(
            [model.lat_dist.gmu_parameters(),
             model.svgp.g0_parameters()]))

    params1 = list(
        itertools.chain.from_iterable([
            model.lat_dist.concentration_parameters(),
            model.lprior.parameters(),
            model.svgp.g1_parameters()
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
        msg = ('\riter {:3d} | elbo {:.3f} | kl {:.3f} | loss {:.3f} |').format(
            i, svgp_elbo_val / Z, kl_val / Z, loss_val / Z)

        print(msg + model.lat_dist.msg(Y, batch_idxs, sample_idxs) +
              model.svgp.msg + model.lprior.msg,
              end="\r")


def fit(dataset: Union[Tensor, DataLoader],
        model: SvgpLvm,
        optimizer=optim.Adam,
        n_mc: int = 32,
        burnin: int = 100,
        lrate: float = 1E-3,
        max_steps: int = 1000,
        stop=None,
        print_every: int = 50,
        mask_Ts=None,
        neuron_idxs: Optional[List[int]] = None,
        prior_m=None,
        analytic_kl=False,
        accumulate_gradient=True):
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
    n = dataloader.n if neuron_idxs is None else len(neuron_idxs)
    m = dataloader.batch_pool_size

    for i in range(max_steps):
        loss_vals, kl_vals, svgp_vals = [], [], []
        ramp = 1 - np.exp(-i / burnin)
        for sample_idxs, batch_idxs, batch in dataloader:
            weight = len(batch_idxs) / m
            
            svgp_elbo, kl = model(batch,
                                  n_mc,
                                  batch_idxs=batch_idxs,
                                  sample_idxs=sample_idxs,
                                  neuron_idxs=neuron_idxs,
                                  m=prior_m,
                                  analytic_kl=analytic_kl)

            loss = (-svgp_elbo) + (ramp * kl)  # -LL
            loss_vals.append(weight*loss.item())
            kl_vals.append(weight*kl.item())
            svgp_vals.append(weight*svgp_elbo.item())

            if accumulate_gradient and (batch_idxs is not None):
                loss *= weight #scale so the total sum of losses is constant

            loss.backward()

            if not accumulate_gradient:
                opt.step()  #update parameters for every batch
                opt.zero_grad()  #reset gradients

        if accumulate_gradient:
            opt.step()  #accumulate gradients across all batches, then update
            opt.zero_grad()  #reset gradients after all batches

        scheduler.step()
        print_progress(model, n, m, n_samples, i, np.sum(loss_vals),
                       np.sum(kl_vals), np.sum(svgp_vals), print_every, batch,
                       None, None)

        # terminate if stop is True
        if stop is not None:
            if stop(model, i, np.sum(loss_vals)):
                break
