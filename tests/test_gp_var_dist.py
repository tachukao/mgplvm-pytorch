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
    lat_dist = mgp.GPBaseLatDist(manif,
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
    n_samples = 5  # number of samples
    ell = 7

    #generate from GPFA generative model
    ts = np.array([np.arange(m) for nsamp in range(n_samples)])[:, None, :]

    dts_sq = (ts[..., None] - ts[..., None, :])**2  #(n_samples x 1 x m x m)
    dts_sq = np.sum(dts_sq, axis=-3)  #(n_samples x m x m)
    K = np.exp(-dts_sq / (2 * ell**2)) + 1e-6 * np.eye(m)[None, ...]
    L = np.linalg.cholesky(K)
    us = np.random.normal(0, 1, size=(m, d))
    xs = L @ us  #(n_samples x m x d)
    print('xs:', xs.shape)
    w = np.random.normal(0, 1, size=(n, d))
    Y = w @ xs.transpose(0, 2, 1)  #(n_samples x n x m)
    Y = Y + np.random.normal(0, 0.2, size=Y.shape)
    print('Y:', Y.shape, np.std(Y), np.quantile(Y, 0.99))

    data = torch.tensor(Y, device=device, dtype=torch.get_default_dtype())

    names = ['Diagonal', 'Circulant']
    for nGP, GP in enumerate([mgp.GPDiagLatDist, mgp.GPCircLatDist]):
        print('\n', names[nGP])

        # specify manifold, kernel and rdist
        manif = mgp.manifolds.Euclid(m, dfit)
        #kernel = mgp.kernels.Linear(n, dfit)
        kernel = mgp.kernels.Linear(n, dfit, ard=True, learn_scale=False, Y=Y)

        lat_dist = GP(manif, m, n_samples, torch.Tensor(ts))
        _scale = torch.ones(n_samples, d, m) * 0.5 * (1 + torch.randn(m) / 100
                                                     )  #n_diag x T
        lat_dist._scale = nn.Parameter(data=inv_softplus(_scale),
                                       requires_grad=True)

        ###construct prior
        prior = mgp.priors.Null(manif)

        # generate model
        likelihood = mgp.likelihoods.Gaussian(n, Y=Y, d=dfit)
        z = manif.inducing_points(n, n_z)
        mod = mgp.models.SvgpLvm(n, m, n_samples, z, kernel, likelihood,
                                 lat_dist, prior).to(device)

        ### test that training runs ###
        n_mc = 16

        mgp.optimisers.svgp.fit(data,
                                mod,
                                optimizer=optim.Adam,
                                n_mc=n_mc,
                                max_steps=50,
                                burnin=50,
                                lrate=10e-2,
                                print_every=20)

        print('lat ell, scale:',
              mod.lat_dist.ell.detach().flatten(),
              mod.lat_dist.scale.detach().mean(0).mean(-1))

        ### also test KL divergence ###
        lat_dist._ell.requires_grad = False
        lat_dist._ell[...] = inv_softplus(lat_dist._ell[...] * 0 + 2)
        print(lat_dist.ell)
        jitter = torch.eye(m).to(device) * 1e-15

        kl = mod.lat_dist.kl()  #computed KL from the class
        Kfull = mod.lat_dist.full_cov().to(
            device)  #computed full covariance (n_samples x d x m x m)
        mu = mod.lat_dist.lat_mu.transpose(-1, -2).to(
            device)  #(n_samples x d x m); posterior mean

        #construct prior
        lhalf = lat_dist.ell / np.sqrt(2)
        sighalf_sq = 1 * 2**(1 / 4) * np.pi**(-1 / 4) * lat_dist.ell**(-1 / 2)
        Khalf_prior = sighalf_sq[..., None] * torch.exp(
            -torch.tensor(dts_sq[:, None, ...]).to(device) /
            (2 * lhalf[..., None]**2))
        Kprior = Khalf_prior @ Khalf_prior  #(n_samples x 1 x m x m)

        #construct pytorch distributions and compute KL from those
        p = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(1).to(device), covariance_matrix=(Kprior + jitter))
        q = torch.distributions.multivariate_normal.MultivariateNormal(
            mu, covariance_matrix=(Kfull + jitter))
        kl_true = torch.distributions.kl.kl_divergence(q, p)

        print('computed:\n', kl)
        print('true:\n', kl_true)

        assert torch.allclose(kl, kl_true)


if __name__ == '__main__':
    test_K_half()
    test_GP_lat_prior()
