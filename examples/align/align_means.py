import numpy as np
import torch
from torch import optim


def generate_binary_array(n, l):
    if n == 0:
        return l
    else:
        if len(l) == 0:
            return generate_binary_array(
                n - 1, [np.array([-1]), np.array([1])])
        else:
            return generate_binary_array(
                n - 1, ([np.concatenate([i, [-1]])
                         for i in l] + [np.concatenate([i, [1]]) for i in l]))


def align_torus(mod, target):
    # target should be mxd
    target = torch.tensor(target)

    def dist(newmus, params):
        mus = mod.lat_dist.manif.gmul(newmus, params)
        loss = mod.lat_dist.manif.distance(mus, target)
        return loss.mean() / mod.lat_dist.m

    mus = mod.lat_dist.manif.prms.data.cpu()
    optloss = np.inf

    for coords in generate_binary_array(mod.lat_dist.d, []):
        coords = torch.tensor(coords).reshape(1, mod.lat_dist.d)
        newmus = coords * mus

        for i in range(5):  # random restarts to avoid local minima
            #params = torch.zeros(mod.lat_dist.d)
            params = torch.rand(mod.lat_dist.d) * 2 * np.pi
            params.requires_grad_()
            optimizer = optim.LBFGS([params])

            def closure():
                optimizer.zero_grad()
                loss = dist(newmus, params)
                loss.backward()
                return loss

            optimizer.step(closure)

            loss = closure()
            print('coordinate system:', coords.numpy(), i, 'loss:',
                  loss.data.cpu().numpy())
            if loss < optloss:
                optloss = loss
                optcoords = coords
                optparams = params.data.cpu()

    newparam = (mod.lat_dist.manif.gmul(optcoords * mus, optparams) +
                2 * np.pi) % (2 * np.pi)
    device = mod.lat_dist.manif.mu.device
    mod.lat_dist.manif.mu.data = newparam.to(device)
    return mod
