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

# Custom Stopping Criterions


class LossMarginStop():
    def __init__(self, loss_margin=0, stop_after=10):
        self.lowest_loss = np.inf
        self.stop_ = 0
        self.loss_margin = loss_margin
        self.stop_after = stop_after

    def __call__(self, model, i, loss_val):
        if loss_val < self.lowest_loss:
            self.lowest_loss = loss_val
        if loss_val <= self.lowest_loss + self.loss_margin:
            self.stop_ = 0
        else:
            self.stop_ += 1
        return (self.stop_ > self.stop_after)


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
            print_sgp_progress(model, i_step, n, m, sgp_elbo, kl, loss)

    return model


def print_sgp_progress(model, i, n, m, sgp_elbo, kl, loss):
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


def sort_params(model, hook, trainGP, svgp=False):
    '''apply burnin period to Sigma_Q and alpha^2
    allow for masking of certain conditions for use in crossvalidation'''

    # parameters to be optimized

    params: List[List[Tensor]] = [[], [], []] if svgp else [[], []]

    for param in model.parameters():
        if (param.shape == model.lat_dist.gamma.shape) and torch.all(
                param == model.lat_dist.gamma):
            param.register_hook(hook)  # option to mask gradients
            params[1].append(param)
        elif (param.shape == model.lat_dist.manif.mu.shape) and torch.all(
                param == model.lat_dist.manif.mu):
            param.register_hook(hook)  # option to mask gradients
            params[0].append(param)
        elif svgp and (param.shape == model.svgp.q_mu.shape) and torch.all(
                param == model.svgp.q_mu):
            params[2].append(param)
        elif svgp and (param.shape == model.svgp.q_sqrt.shape) and torch.all(
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


def svgp(Y,
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
         hook=None,
         neuron_idxs=None,
         loss_margin=0.,
         stop_iters=10):
    '''
    max_steps [optional]: int, default=1000
        maximum number of training iterations
    
    batch_pool [optional] : None or int list
        pool of indices from which to batch (used to train a partial model)
        
    loss_margin [optional] : float, default=0
        loss margin tolerated for training progress
        
    stop_iters [optional]: int, default=10
        maximum number of training iterations above loss margin tolerated before stopping
    '''

    n, m, _ = Y.shape  # neurons, conditions, samples
    data = torch.from_numpy(Y).float().to(device)
    ts = ts if ts is None else ts.to(device)
    data_size = m  #total conditions

    def generate_batch_idxs(batch_pool=None):
        if batch_pool is None:
            idxs = np.arange(data_size)
        else:
            idxs = copy.copy(batch_pool)
            data_size = len(idxs)
        if model.lprior.name == "Brownian":
            # if prior is Brownian, then batches have to be contiguous

            i0 = np.random.randint(1, data_size - 1)
            if i0 < batch_size / 2:
                batch_idxs = idxs[:int(round(batch_size / 2 + i0))]
            elif i0 > (data_size - batch_size / 2):
                batch_idxs = idxs[int(round(i0 - batch_size / 2)):]
            else:
                batch_idxs = idxs[int(round(i0 - batch_size /
                                            2)):int(round(i0 +
                                                          batch_size / 2))]
            #print(len(batch_idxs))
            return batch_idxs

            #start = np.random.randint(data_size - batch_size)
            #return idxs[start:start + batch_size]
        else:
            np.random.shuffle(idxs)
            return idxs[0:batch_size]

    #optionally mask some time points
    if hook is None:

        def hook(grad):
            return grad

    params = sort_params(model, hook, True, svgp=True)
    opt = optimizer(params[0], lr=lrate)  # instantiate optimizer
    opt.add_param_group({'params': params[1]})
    opt.add_param_group({'params': params[2]})

    # set learning rate schedule so sigma updates have a burn-in period
    def fburn(x):
        return 1 - np.exp(-x / (3 * burnin))

    LRfuncs = [lambda x: 1, fburn, lambda x: 1]
    scheduler = LambdaLR(opt, lr_lambda=LRfuncs)

    lowest_loss = np.inf
    stop_ = 0

    for i in range(max_steps):
        opt.zero_grad()
        ramp = 1 - np.exp(-i / burnin)  # ramp the entropy

        if (batch_size is None and batch_pool is None):
            batch_idxs = None
        elif batch_size is None:
            batch_idxs = batch_pool
            m = len(batch_idxs)
        else:
            batch_idxs = generate_batch_idxs(batch_pool=batch_pool)
            m = len(batch_idxs)  #use for printing likelihoods etc.

        svgp_elbo, kl = model(data,
                              n_mc,
                              batch_idxs=batch_idxs,
                              ts=ts,
                              neuron_idxs=neuron_idxs)

        loss = (-svgp_elbo) + (ramp * kl)  # -LL
        loss_val = loss.item()
        # terminate if stop is True
        if stop is not None:
            if stop(model, i, loss_val): break
        loss.backward()
        opt.step()
        scheduler.step()

        if loss_val < lowest_loss:
            lowest_loss = loss_val

        if loss_val <= lowest_loss + loss_margin:
            stop_ = 0
        else:
            stop_ += 1

        if i % print_every == 0 or stop_ > stop_iters:
            mu_mag = np.mean(
                np.sqrt(
                    np.sum(model.lat_dist.manif.prms.data.cpu().numpy()[:]**2,
                           axis=1)))
            sig = np.median(
                np.concatenate([
                    np.diag(sig)
                    for sig in model.lat_dist.prms.data.cpu().numpy()
                ]))
            msg = ('\riter {:3d} | elbo {:.3f} | kl {:.3f} | loss {:.3f} ' +
                   '| |mu| {:.3f} | sig {:.3f} |').format(
                       i,
                       svgp_elbo.item() / (n * m),
                       kl.item() / (n * m), loss_val / (n * m), mu_mag, sig)
            print(msg + model.kernel.msg + model.lprior.msg, end="\r")

    return model
