from __future__ import print_function
import numpy as np
import mgplvm
from mgplvm.utils import softplus
from mgplvm.manifolds import Torus, Euclid, So3
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR


def sgp(Y,
        model,
        device,
        optimizer=optim.Adam,
        outdir="results/vanilla",
        max_steps=1000,
        n_b=128,
        burnin='default',
        lrate=1e-3,
        print_every=10,
        callback=None,
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
    if type(model) == mgplvm.models.SgpComb:
        params = sort_params_prod(model, _Tlearn_hook, trainGP)
    else:
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
    i_step = -1
    Ls = np.zeros(10)
    while (i_step < max_steps) and (np.std(Ls) > sigma_thresh or i_step <
                                    (2.5 * burnin)):
        i_step += 1
        optimizer.zero_grad()
        ramp = 1 - np.exp(-i_step / burnin)  # ramp the entropy

        for _ in range(nbatch):
            sgp_elbo, kl = model(data, n_b)  # log p(Y|G), KL(Q(G), p(G))
            loss = (-sgp_elbo + (ramp * kl))  # -LL
            loss.backward()

        optimizer.step()
        scheduler.step()
        if i_step % print_every == 0:
            print_progress(model, i_step, n, m, sgp_elbo, kl, loss)
        if callback is not None:
            callback(model, i_step)

        Ls[i_step % 10] = loss.item() / (n * m)

    return model


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

    if type(model) == mgplvm.models.SgpComb:
        sigs = [r.prms.data.cpu().numpy() for r in model.rdist]
        sigs = [np.concatenate([np.diag(s) for s in sig]) for sig in sigs]
        sig = np.median(np.concatenate(sigs))
        alpha_mag = torch.stack([p[0] for p in model.kernel.prms
                                 ]).mean().data.cpu().numpy()
        ell_mag = torch.stack([p[1] for p in model.kernel.prms
                               ]).mean().data.cpu().numpy()
    else:
        sig = np.median(
            np.concatenate(
                [np.diag(sig) for sig in model.rdist.prms.data.cpu().numpy()]))
        alpha_mag, ell_mag = [
            val.mean().data.cpu().numpy() for val in model.kernel.prms
        ]

    if model.lprior.name == "Brownian":
        brownian_c, brownian_eta = model.lprior.prms
        print((
            '\riter {:4d} | elbo {:.4f} | kl {:.4f} | loss {:.4f} ' +
            '| |mu| {:.4f} | alpha_sqr {:.4f} | ell {:.4f} | sig {:.4f} | brownian_c {:.4f} | brownian_eta {:.4f}'
        ).format(i,
                 sgp_elbo.item() / (n * m),
                 kl.item() / (n * m),
                 loss.item() / (n * m), mu_mag, alpha_mag**2, ell_mag, sig,
                 brownian_c.item(), brownian_eta.item()),
              end='\r')
    else:
        print(('\riter {:4d} | elbo {:.4f} | kl {:.4f} | loss {:.4f} ' +
               '| |mu| {:.4f} | alpha_sqr {:.4f} | ell {:.4f} | sig {:.8f}'
               ).format(i,
                        sgp_elbo.item() / (n * m),
                        kl.item() / (n * m),
                        loss.item() / (n * m), mu_mag, alpha_mag**2, ell_mag,
                        sig),
              end='\r')


def sort_params(model, hook, trainGP, svgp=False):
    '''apply burnin period to Sigma_Q and alpha^2
    allow for masking of certain conditions for use in crossvalidation'''

    # parameters to be optimized

    if svgp:
        params = [[], [], []]
    else:
        params = [[], []]

    for param in model.parameters():
        if (param.shape == model.rdist.gamma.shape) and torch.all(
                param == model.rdist.gamma):
            param.register_hook(hook)  # option to mask gradients
            params[1].append(param)
        elif (param.shape == model.manif.mu.shape) and torch.all(
                param == model.manif.mu):
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
            if (param.shape == model.kernel.ell.shape) and torch.all(
                    param == model.kernel.ell):
                params[1].append(param)
            else:
                params[0].append(param)

    return params


def sort_params_prod(model, hook, trainGP):
    '''apply burnin period to Sigma_Q and alpha^2
    allow for masking of certain conditions for use in crossvalidation'''

    # parameters to be optimized

    params = [[], []]
    for ref_dist in model.rdist:  # variational widths
        param = ref_dist.gamma
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
         outdir="results/vanilla",
         max_steps=1000,
         n_mc=128,
         burnin=100,
         lrate=1E-3,
         callback=None,
         print_every=50,
         batch_size=None):
    n, m, _ = Y.shape  # neurons, conditions, samples
    data = torch.from_numpy(Y).float().to(device)
    data_size = m

    def generate_batch_idxs():
        idxs = np.arange(data_size)
        if model.lprior.name == "Brownian":
            # if prior is Brownian, then batches have to be contiguous
            start = np.random.randint(data_size - batch_size)
            return idxs[start:start + batch_size]
        else:
            np.shuffle(idxs)
            return idxs[0:batch_size]

    #opt = optimizer(model.parameters(), lr=lrate)

    def _Tlearn_hook(grad):
        return grad

    params = sort_params(model, _Tlearn_hook, True, svgp=True)
    opt = optimizer(params[0], lr=lrate)  # instantiate optimizer
    opt.add_param_group({'params': params[1]})
    opt.add_param_group({'params': params[2]})

    # set learning rate schedule so sigma updates have a burn-in period
    def fburn(x):
        if x < 50:
            return 0
        #only optimize
        x -= 50
        return 1 - np.exp(-x / (3 * burnin))

    def finit(x):
        if x < 50:
            return 0
        return 1

    LRfuncs = [finit, fburn, lambda x: 1]
    scheduler = LambdaLR(opt, lr_lambda=LRfuncs)

    for i in range(max_steps):  # come up with a different stopping condition
        opt.zero_grad()
        ramp = 1 - np.exp(-i / burnin)  # ramp the entropy

        if batch_size is None:
            svgp_lik, svgp_kl, kl = model(data, n_mc, batch_idxs=None)
        else:
            batch_idxs = generate_batch_idxs()
            svgp_lik, svgp_kl, kl = model(data, n_mc, batch_idxs=batch_idxs)

        svgp_elbo = svgp_lik - svgp_kl
        loss = (-svgp_elbo) + (ramp * kl)  # -LL
        loss.backward()
        opt.step()
        scheduler.step()
        if i % print_every == 0:
            mu_mag = np.mean(
                np.sqrt(
                    np.sum(model.manif.prms.data.cpu().numpy()[:]**2, axis=1)))
            alpha_mag, ell_mag = [
                np.mean(val.data.cpu().numpy()) for val in model.kernel.prms
            ]
            print((
                '\riter {:4d} | elbo {:.4f} | svgp_kl {:.4f} | kl {:.4f} | loss {:.4f} '
                + '| |mu| {:.4f} | alpha_sqr {:.4f} | ell {:.4f}').format(
                    i,
                    svgp_elbo.item() / (n * m),
                    svgp_kl.item() / (n * m),
                    kl.item() / (n * m),
                    loss.item() / (n * m), mu_mag, alpha_mag**2, ell_mag),
                  end='\r')

        if callback is not None:
            callback(model, i)

    return model