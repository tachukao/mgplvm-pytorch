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
    params: List[List[Tensor]] = [[], [], []]
    for param in model.parameters():
        if (param.shape == lat_params[1].shape) and torch.all(
                param == lat_params[1]):
            param.register_hook(hook)  # option to mask gradients
            params[1].append(param)
        elif (param.shape == lat_params[0].shape) and torch.all(
                param == lat_params[0]):
            param.register_hook(hook)  # option to mask gradients
            params[0].append(param)
        elif (param.shape == model.svgp.q_mu.shape) and torch.all(
                param == model.svgp.q_mu):
            params[2].append(param)
        elif (param.shape == model.svgp.q_sqrt.shape) and torch.all(
                param == model.svgp.q_sqrt):
            params[2].append(param)
        elif trainGP:  # only update GP parameters if trainGP
            # add ell to group 2
            if (('QuadExp' in model.kernel.name)
                    and (param.shape == model.kernel.ell.shape)
                    and torch.all(param == model.kernel.ell)):
                params[1].append(param)
            else:
                params[0].append(param)

    return params


def sort_params_prod(model, hook, trainGP):
    '''apply burnin period to Sigma_Q and alpha^2
    allow for masking of certain conditions for use in crossvalidation'''

    # parameters to be optimized

    params = [[], []]
    for lat_dist in model.lat_dist:  # variational widths
        param = lat_dist.gamma
        param.register_hook(hook)
        params[1].append(param)
    for manif in model.manif:  # varriational maens
        param = manif.mu
        param.register_hook(hook)
        params[0].append(param)
    for kernel in model.kernel.kernels:  # kerernels
        params[0].append(kernel.ell)
        params[0].append(kernel.alpha)
    for z in model.z:  # inducingg points
        params[0].append(z.z)

    params[0].append(model.sgp.sigma)  # noise variance

    return params


def print_progress(model,
                   n,
                   m,
                   i,
                   loss_val,
                   kl_val,
                   svgp_elbo_val,
                   print_every=50):
    if i % print_every == 0:
        mu_mag = np.mean(
            np.sqrt(
                np.sum(model.lat_dist.prms[0].data.cpu().numpy()[:]**2,
                       axis=1)))
        sig = np.median(
            np.concatenate([
                np.diag(sig)
                for sig in model.lat_dist.prms[1].data.cpu().numpy()
            ]))
        msg = ('\riter {:3d} | elbo {:.3f} | kl {:.3f} | loss {:.3f} ' +
               '| |mu| {:.3f} | sig {:.3f} |').format(i,
                                                      svgp_elbo_val / (n * m),
                                                      kl_val / (n * m),
                                                      loss_val / (n * m),
                                                      mu_mag, sig)
        print(msg + model.kernel.msg + model.lprior.msg, end="\r")


def generate_batch_idxs(model, data_size, batch_pool=None):
    if batch_pool is None:
        idxs = np.arange(data_size)
    else:
        idxs = copy.copy(batch_pool)
    if model.lprior.name == "Brownian":
        # if prior is Brownian, then batches have to be contiguous
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

        #start = np.random.randint(data_size - batch_size)
        #return idxs[start:start + batch_size]
    else:
        np.random.shuffle(idxs)
        return idxs[0:batch_size]


def optimise(Y,
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
             n_svgp=0,
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
    def fburn(x, burnin=100):
        return 1 - np.exp(-x / (3 * burnin))

    n, m, _ = Y.shape  # neurons, conditions, samples
    data = torch.from_numpy(Y).float().to(device)
    ts = ts if ts is None else ts.to(device)
    data_size = m if batch_pool is None else len(batch_pool)  #total conditions
    n = n if neuron_idxs is None else len(neuron_idxs)
    #optionally mask some time points
    mask_Ts = mask_Ts if mask_Ts is not None else lambda x: x

    params = sort_params(model, mask_Ts, True)
    opt = optimizer(params[0], lr=lrate)  # instantiate optimizer
    opt.add_param_group({'params': params[1]})
    opt.add_param_group({'params': params[2]})

    LRfuncs = [lambda x: 1, fburn, lambda x: 1]
    scheduler = LambdaLR(opt, lr_lambda=LRfuncs)

    for i in range(max_steps):
        opt.zero_grad()
        ramp = fburn(i)

        if (batch_size is None and batch_pool is None):
            batch_idxs = None
        elif batch_size is None:
            batch_idxs = batch_pool
            m = len(batch_idxs)
        else:
            batch_idxs = generate_batch_idxs(model,
                                             data_size,
                                             batch_pool=batch_pool)
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
                       print_every)

    return model
