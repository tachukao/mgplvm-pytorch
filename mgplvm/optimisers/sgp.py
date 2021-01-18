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


def sort_params(model, hook, trainGP):
    '''apply burnin period to Sigma_Q and alpha^2
    allow for masking of certain conditions for use in crossvalidation'''

    # parameters to be optimized

    lat_params = list(model.lat_dist.parameters())
    params: List[List[Tensor]] = [[], []]
    for param in model.parameters():
        if (param.shape == lat_params[1].shape) and torch.all(
                param == lat_params[1]):
            param.register_hook(hook)  # option to mask gradients
            params[1].append(param)
        elif (param.shape == lat_params[0].shape) and torch.all(
                param == lat_params[0]):
            param.register_hook(hook)  # option to mask gradients
            params[0].append(param)
        elif trainGP:  # only update GP parameters if trainGP
            # add ell to group 2
            if (('QuadExp' in model.kernel.name)
                    and (param.shape == model.kernel.ell.shape)
                    and torch.all(param == model.kernel.ell)):
                params[1].append(param)
            else:
                params[0].append(param)

    return params


def print_progress(model, i, n, m, sgp_elbo, kl, loss):
    if type(model.manif) == So3:
        mu_mag = np.mean(
            np.sqrt(
                np.sum(model.manif.prms.data.cpu().numpy()[:, 1:]**2, axis=1)))
    elif type(model.manif) == list:
        ms = [m.prms.data.cpu().numpy() for m in model.manif]
        mu_mag = np.mean([np.mean(np.sqrt(np.sum(m**2, axis=1))) for m in ms])
    else:
        mu_mag = np.mean(
            np.sqrt(np.sum(model.manif.prms.data.cpu().numpy()**2, axis=1)))

    #if type(model) == mgplvm.models.SgpComb:
    #    sigs = [r.prms.data.cpu().numpy() for r in model.rdist]
    #    sigs = [np.concatenate([np.diag(s) for s in sig]) for sig in sigs]
    #    sig = np.median(np.concatenate(sigs))
    #    alpha_mag = torch.stack([p[0] for p in model.kernel.prms
    #                             ]).mean().data.cpu().numpy()
    #    ell_mag = torch.stack([p[1] for p in model.kernel.prms
    #                           ]).mean().data.cpu().numpy()
    sig = np.median(
        np.concatenate(
            [np.diag(sig) for sig in model.rdist.prms.data.cpu().numpy()]))
    alpha_mag, ell_mag = [
        val.mean().data.cpu().numpy() for val in model.kernel.prms
    ]

    msg = (
        '\riter {:4d} | elbo {:.4f} | kl {:.4f} | loss {:.4f} | |mu| {:.4f} | alpha_sqr {:.4f} | ell {:.4f}'
    ).format(i,
             sgp_elbo.item() / (n * m),
             kl.item() / (n * m),
             loss.item() / (n * m), mu_mag, alpha_mag**2, ell_mag)
    print(msg + " | " + model.lprior.msg, "\r")


def sgp(Y,
        model,
        device,
        optimizer=optim.Adam,
        n_b=128,
        burnin='default',
        lrate=1e-3,
        print_every=10,
        max_steps=1000,
        stop=None,
        trainGP=True,
        nbatch=1,
        Tfix=slice(0),
        sigma_thresh=0.0001):
    def _Tlearn_hook(grad):
        ''' used to 'mask' some gradients for cv'''
        grad[Tfix, ...] *= 0
        return grad

    if burnin == 'default':
        burnin = 5 / lrate
    n, m, _ = Y.shape  # neurons, conditions, samples
    data = torch.tensor(Y, dtype=torch.get_default_dtype()).to(device)

    # parameters to be optimized
    #if type(model) == mgplvm.models.SgpComb:
    #    params = sort_params_prod(model, _Tlearn_hook, trainGP)
    params = sort_params(model, _Tlearn_hook, trainGP)
    optimizer = optimizer(params[0], lr=lrate)  # instantiate optimizer
    optimizer.add_param_group({'params': params[1]})

    # set learning rate schedule so sigma updates have a burn-in period
    def fburn(x):
        return 1 - np.exp(-x / (3 * burnin))

    LRfuncs = [lambda x: 1, fburn]
    scheduler = LambdaLR(optimizer, lr_lambda=LRfuncs)

    if nbatch > 1:
        n_b = int(round(n_b / nbatch))

    for i_step in range(max_steps):
        optimizer.zero_grad()
        ramp = 1 - np.exp(-i_step / burnin)  # ramp the entropy

        loss_val = 0
        for _ in range(nbatch):
            sgp_elbo, kl = model(data, n_b)  # log p(Y|G), KL(Q(G), p(G))
            loss = (-sgp_elbo + (ramp * kl))  # -LL
            loss.backward()
            loss_val = loss.item() + loss_val

        if stop is not None:
            if stop(model, i_step, loss_val): break
        optimizer.step()
        scheduler.step()
        if i_step % print_every == 0:
            print_progress(model, i_step, n, m, sgp_elbo, kl, loss)

    return model
