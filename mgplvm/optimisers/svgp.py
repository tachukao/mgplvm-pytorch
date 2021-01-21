from __future__ import print_function
import abc
import numpy as np
import mgplvm
from mgplvm.utils import softplus
from mgplvm.manifolds import Torus, Euclid, So3
import torch
from torch import optim, Tensor
from torch.optim.lr_scheduler import LambdaLR
from typing import List
import itertools


def sort_params(model, hook):
    '''apply burnin period to Sigma_Q and alpha^2
    allow for masking of certain conditions for use in crossvalidation'''

    for prm in model.lat_dist.parameters():
        prm.register_hook(hook)

    params0 = list(
        itertools.chain.from_iterable([
            model.z.parameters(),
            model.likelihood.parameters(),
            model.lprior.parameters(),
            model.lat_dist.gmu_parameters(),
            model.kernel.parameters(),
            [model.svgp.q_mu, model.svgp.q_sqrt],
        ]))

    params1 = list(
        itertools.chain.from_iterable(
            [model.lat_dist.concentration_parameters()]))

    params = [params0, params1]
    return params


def print_progress(model,
                   n,
                   m,
                   i,
                   loss_val,
                   kl_val,
                   svgp_elbo_val,
                   print_every=50,
                   Y=None,
                   batch_idxs=None):
    if i % print_every == 0:
        mu = model.lat_dist.lat_gmu(Y, batch_idxs).data.cpu().numpy()
        gamma = model.lat_dist.lat_gamma(Y, batch_idxs).diagonal(
            dim1=-1, dim2=-2).data.cpu().numpy()
        mu_mag = np.mean(np.sqrt(gamma))
        sig = np.median(np.concatenate([np.diag(sig) for sig in gamma]))
        msg = ('\riter {:3d} | elbo {:.3f} | kl {:.3f} | loss {:.3f} ' +
               '| |mu| {:.3f} | sig {:.3f} |').format(i,
                                                      svgp_elbo_val / (n * m),
                                                      kl_val / (n * m),
                                                      loss_val / (n * m),
                                                      mu_mag, sig)
        print(msg + model.kernel.msg + model.lprior.msg, end="\r")


def generate_batch_idxs(model, data_size, batch_pool=None, batch_size=None):
    if batch_pool is None:
        idxs = np.arange(data_size)
    else:
        idxs = batch_pool
    if model.lprior.name in ["Brownian", "ARP"]:
        # if prior is Brownian or ARP, then batches have to be contiguous
        i0 = np.random.randint(1, data_size - 1)
        if i0 < batch_size / 2:
            batch_idxs = idxs[:int(round(batch_size / 2 + i0))]
        elif i0 > (data_size - batch_size / 2):
            batch_idxs = idxs[int(round(i0 - batch_size / 2)):]
        else:
            batch_idxs = idxs[int(round(i0 - batch_size /
                                        2)):int(round(i0 + batch_size / 2))]
        #print(len(batch_idxs))
        return batch_idxs
    else:
        if batch_size is None:
            return idxs
        else:
            return np.random.choice(idxs, size=batch_size, replace=False)


def fit(Y,
        model,
        device,
        optimizer=optim.Adam,
        n_mc=128,
        burnin=100,
        lrate=1E-3,
        max_steps=1000,
        stop=None,
        print_every=50,
        batch_size=None,
        ts=None,
        batch_pool=None,
        mask_Ts=None,
        neuron_idxs=None):
    '''
    max_steps : Optional[int], default=1000
        maximum number of training iterations
    
    batch_pool : Optional[int list]
        pool of indices from which to batch (used to train a partial model)
    '''

    # set learning rate schedule so sigma updates have a burn-in period
    def fburn(x):
        return 1 - np.exp(-x / (3 * burnin))

    if len(Y.shape) > 2:
        _, n, m = Y.shape  # samples, neurons, conditions
    else:
        n, m = Y.shape  # neuron x conditions
    data = torch.tensor(Y, dtype=torch.get_default_dtype()).to(device)
    ts = ts if ts is None else ts.to(device)
    data_size = m if batch_pool is None else len(batch_pool)  #total conditions
    n = n if neuron_idxs is None else len(neuron_idxs)
    #optionally mask some time points
    mask_Ts = mask_Ts if mask_Ts is not None else lambda x: x

    params = sort_params(model, mask_Ts)

    # instantiate optimizer
    opt = optimizer([
        {
            'params': params[0]
        },
        {
            'params': params[1]
        },
    ], lr=lrate)

    scheduler = LambdaLR(opt, lr_lambda=[lambda x: 1, fburn])

    for i in range(max_steps):
        opt.zero_grad()
        ramp = 1 - np.exp(-i / burnin)

        if (batch_size is None and batch_pool is None):
            batch_idxs = None
        elif batch_size is None:
            batch_idxs = batch_pool
            m = len(batch_idxs)
        else:
            batch_idxs = generate_batch_idxs(model,
                                             data_size,
                                             batch_pool=batch_pool,
                                             batch_size=batch_size)
            m = len(batch_idxs)  #use for printing likelihoods etc.

        svgp_elbo, kl = model(data,
                              n_mc,
                              batch_idxs=batch_idxs,
                              ts=ts,
                              neuron_idxs=neuron_idxs)

        loss = (-svgp_elbo) + (ramp * kl)  # -LL
        loss_val = loss.item()
        kl_val = kl.item()
        svgp_elbo_val = svgp_elbo.item()
        # terminate if stop is True
        if stop is not None:
            if stop(model, i, loss_val): break
        loss.backward()
        opt.step()
        scheduler.step()
        print_progress(model, n, m, i, loss_val, kl_val, svgp_elbo_val,
                       print_every, data, batch_idxs)

    return model
