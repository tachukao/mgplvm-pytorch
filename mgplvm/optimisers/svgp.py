from __future__ import print_function
import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from .data import NeuralDataLoader
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
        mu = lat_dist.lat_gmu(Y,
                              batch_idxs=batch_idxs,
                              sample_idxs=sample_idxs)
        gamma = lat_dist.lat_gamma(Y,
                                   batch_idxs=batch_idxs,
                                   sample_idxs=sample_idxs).diagonal(dim1=-1,
                                                                     dim2=-2)

        mu_mag = torch.sqrt(torch.mean(mu**2)).item()
        sig = torch.median(gamma).item()
        msg = ('\riter {:3d} | elbo {:.3f} | kl {:.3f} | loss {:.3f} ' +
               '| |mu| {:.3f} | sig {:.3f} |').format(i, svgp_elbo_val / Z,
                                                      kl_val / Z, loss_val / Z,
                                                      mu_mag, sig)
        print(msg + model.kernel.msg + model.lprior.msg, end="\r")


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
        mask_Ts=None,
        neuron_idxs=None,
        batch_size=None,
        batch_pool=None,
        sample_size=None,
        sample_pool=None):
    '''
    Parameters
    ----------
    Y : np.array
        data matrix of dimensions (n_samples x n x m)
    device : torch.device
        torch device
    max_steps : Optional[int], default=1000
        maximum number of training iterations
    batch_pool : Optional[int list]
        pool of indices from which to batch (used to train a partial model)
    '''

    # set learning rate schedule so sigma updates have a burn-in period
    def fburn(x):
        return 1 - np.exp(-x / (3 * burnin))

    if len(Y.shape) > 2:
        n_samples, n, m = Y.shape  # samples, neurons, conditions
    else:
        n, m = Y.shape  # neuron x conditions
        n_samples = 1
    data = torch.tensor(Y, dtype=torch.get_default_dtype()).to(device)
    data_size = m if batch_pool is None else len(batch_pool)  #total conditions
    n = n if neuron_idxs is None else len(neuron_idxs)
    #optionally mask some time points
    mask_Ts = mask_Ts if mask_Ts is not None else lambda x: x

    params = sort_params(model, mask_Ts)

    # instantiate optimizer
    opt = optimizer(params, lr=lrate)

    scheduler = LambdaLR(opt, lr_lambda=[lambda x: 1, fburn])

    data_loader = NeuralDataLoader(data,
                                   sample_size=sample_size,
                                   sample_pool=sample_pool,
                                   batch_size=batch_size,
                                   batch_pool=batch_pool)

    for i in range(max_steps):
        for sample_idxs, batch_idxs, batch in data_loader:
            opt.zero_grad()
            ramp = 1 - np.exp(-i / burnin)

            svgp_elbo, kl = model(batch,
                                  n_mc,
                                  batch_idxs=batch_idxs,
                                  sample_idxs=sample_idxs,
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
            print_progress(model, n, m, n_samples, i, loss_val, kl_val,
                           svgp_elbo_val, print_every, batch, batch_idxs,
                           sample_idxs)
