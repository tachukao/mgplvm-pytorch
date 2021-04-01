import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim, nn
import mgplvm as mgp
from sklearn.cross_decomposition import CCA
from mgplvm.utils import inv_softplus
from mgplvm.fast_utils import sym_toeplitz

torch.set_default_dtype(torch.float64)
device = mgp.utils.get_device()


def test_K_half():
    """check that our implementation of Khalf for the RBF kernel is correct modulo boundary conditions"""
    n_samples, m, dfit = 1, 200, 1

    ###generate K###
    ell = 10
    ts = np.array([np.arange(m) for nsamp in range(n_samples)])[:, None, :]
    dts_sq = (ts[..., None] - ts[..., None, :])**2  #(n_samples x 1 x m x m)
    dts_sq = np.sum(dts_sq, axis=-3)  #(n_samples x m x m)
    K = torch.tensor(np.exp(-dts_sq / (2 * ell**2)))[0, ...]  #n_samples x m x m

    manif = mgp.manifolds.Euclid(m, dfit)
    #initialize with ground truth parameters
    lat_dist = mgp.rdist.GPbase(manif,
                                m,
                                n_samples,
                                torch.Tensor(ts),
                                ell=ell,
                                _scale=1)

    K_half = lat_dist.K_half(None).detach().cpu()  # (n_samples x d x m)
    K_half = sym_toeplitz(K_half[0, 0, :])  #m x m
    K_num = K_half @ K_half  #m x m

    assert torch.allclose(K[90:110, 90:110], K_num[90:110, 90:110])

    return


def test_GP_lat_prior():
    device = mgp.utils.get_device("cuda")  # get_device("cpu")
    d = 2  # dims of latent space
    dfit = 2  #dimensions of fitted space
    n = 50  # number of neurons
    m = 80  # number of conditions / time points
    n_z = 15  # number of inducing points
    n_samples = 10  # number of samples
    Poisson = False

    #generate from GPFA generative model
    ts = np.array([np.arange(m) for nsamp in range(n_samples)])[:, None, :]

    dts_sq = (ts[..., None] - ts[..., None, :])**2  #(n_samples x 1 x m x m)
    dts_sq = np.sum(dts_sq, axis=-3)  #(n_samples x m x m)
    K = np.exp(-dts_sq / (2 * 7**2)) + 1e-6 * np.eye(m)[None, ...]
    L = np.linalg.cholesky(K)
    us = np.random.normal(0, 1, size=(m, d))
    xs = L @ us  #(n_samples x m x d)
    print('xs:', xs.shape)
    w = np.random.normal(0, 1, size=(n, d))
    Y = w @ xs.transpose(0, 2, 1)  #(n_samples x n x m)
    if Poisson:
        Y = np.random.poisson(2 * (Y - np.amin(Y)))
    else:
        Y = Y + np.random.normal(0, 0.2, size=Y.shape)
    print('Y:', Y.shape, np.std(Y), np.quantile(Y, 0.99))

    data = torch.tensor(Y, device=device, dtype=torch.get_default_dtype())

    names = ['Diagonal', 'Circulant']
    for nGP, GP in enumerate([mgp.rdist.GP_diag, mgp.rdist.GP_circ]):
        print(names[nGP])

        # specify manifold, kernel and rdist
        manif = mgp.manifolds.Euclid(m, dfit)
        #kernel = mgp.kernels.Linear(n, dfit)
        kernel = mgp.kernels.Linear(n,
                                    dfit,
                                    ard=True,
                                    learn_scale=False,
                                    Y=Y,
                                    Poisson=Poisson)

        lat_dist = GP(manif, m, n_samples, torch.Tensor(ts))
        _scale = torch.ones(n_samples, d, m) * .2 * (1 + torch.randn(m) / 100
                                                    )  #n_diag x T
        lat_dist._scale = nn.Parameter(data=inv_softplus(_scale),
                                       requires_grad=True)

        ###construct prior
        lprior = mgp.lpriors.Null(manif)

        # generate model
        if Poisson:
            likelihood = mgp.likelihoods.NegativeBinomial(n, Y=Y)
            #likelihood = mgp.likelihoods.Poisson(n)
        else:
            likelihood = mgp.likelihoods.Gaussian(n, Y=Y, d=dfit)
        z = manif.inducing_points(n, n_z)
        mod = mgp.models.SvgpLvm(n, m, n_samples, z, kernel, likelihood,
                                 lat_dist, lprior).to(device)

        print(mod.lat_dist.name)
        ### test that training runs ###
        n_mc = 16

        def cb(mod, i, loss):
            if i % 5 == 0:
                print('')
            return

        mgp.optimisers.svgp.fit(data,
                                mod,
                                optimizer=optim.Adam,
                                n_mc=n_mc,
                                max_steps=5,
                                burnin=50,
                                lrate=10e-2,
                                print_every=5,
                                stop=cb,
                                analytic_kl=True)

        print('lat ell, scale:',
              mod.lat_dist.ell.detach().flatten(),
              mod.lat_dist.scale.detach().mean(0).mean(-1))


if __name__ == '__main__':
    test_K_half()
    test_GP_lat_prior()
