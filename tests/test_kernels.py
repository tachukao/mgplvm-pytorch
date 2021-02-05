import torch
from mgplvm.kernels import QuadExp, Linear, Matern
import numpy as np
from torch import optim
import mgplvm
from mgplvm import rdist, models, optimisers, syndata, likelihoods, lpriors
from mgplvm.manifolds import Torus, Euclid
import sklearn.gaussian_process.kernels as sklkernels
torch.set_default_dtype(torch.float64)


def test_quad_exp_hyp_prms_dims():
    n = 10
    kernel = QuadExp(n, Euclid.distance)
    scale, ell = kernel.prms
    assert (scale.shape == (n,))
    assert (ell.shape == (n,))


def test_quad_exp_trK():
    n_b = 2
    n = 10
    m = 20
    d = 3
    kernel = QuadExp(n, Euclid.distance)
    x = torch.randn(n_b, n, d, m)
    trK1 = kernel.trK(x)
    trK2 = torch.diagonal(kernel(x, x), dim1=2, dim2=3).sum(-1)
    assert torch.allclose(trK1, trK2)


def test_kernels_diagK():
    n_b = 2
    n = 10
    m = 20
    d = 3
    dists = [Euclid.distance, Euclid.distance]
    for i, kerneltype in enumerate([QuadExp, Matern, Linear]):
        if kerneltype is Linear:
            kernel = kerneltype(n, d)
        else:
            kernel = kerneltype(n, dists[i])
        x = torch.randn(n_b, n, d, m)
        diagK1 = kernel.diagK(x)
        diagK2 = torch.diagonal(kernel(x, x), dim1=2, dim2=3)
        assert torch.allclose(diagK1, diagK2)


def test_kernels_run():
    device = mgplvm.utils.get_device()
    d = 1  # dims of latent space
    n = 10  # number of neurons
    m = 25  # number of conditions / time points
    n_z = 15  # number of inducing points
    n_samples = 2  # number of samples
    l = float(0.55 * np.sqrt(d))
    gen = syndata.Gen(syndata.Euclid(d),
                      n,
                      m,
                      variability=0.15,
                      l=l,
                      sigma=0.8,
                      beta=0.1,
                      n_samples=n_samples)
    sig0 = 1.5
    Y = gen.gen_data(ell=25, sig=1)
    data = torch.tensor(Y, dtype=torch.get_default_dtype(), device=device)
    kernels = [
        QuadExp(n, Euclid.distance),
        Linear(n, d),
        # Matern(n, Euclid.distance)
    ]
    for kernel in kernels:
        # specify manifold, kernel and rdist
        manif = Euclid(m, d)
        lat_dist = mgplvm.rdist.ReLie(manif,
                                      m,
                                      n_samples,
                                      initialization='random')
        # generate model
        lik = likelihoods.Gaussian(n)
        lprior = lpriors.Uniform(manif)
        z = manif.inducing_points(n, n_z)
        mod = models.SvgpLvm(n,
                             m,
                             n_samples,
                             z,
                             kernel,
                             lik,
                             lat_dist,
                             lprior,
                             whiten=True).to(device)

        ### test that training runs ###
        trained_mod = optimisers.svgp.fit(data,
                                          mod,
                                          optimizer=optim.Adam,
                                          max_steps=5,
                                          burnin=100,
                                          n_mc=64,
                                          lrate=10E-2,
                                          print_every=50)

    return


def test_quad_exp_kernel():
    n = 1
    d = 3
    mx = 5
    my = 4
    x = np.random.randn(mx, d)
    y = np.random.randn(my, d)
    K = sklkernels.RBF(1.0)(x, y)
    scale = np.ones((n))
    ell = np.ones((n))
    kernel = QuadExp(n, Euclid.distance, scale=scale, ell=ell)
    x_ = torch.tensor(x).transpose(-1, -2)
    y_ = torch.tensor(y).transpose(-1, -2)
    K_ = kernel(x_, y_)[0]
    assert np.allclose(K_.data.cpu().numpy(), K)


def test_quad_exp_kernel():
    n = 1
    d = 3
    mx = 5
    my = 4
    x = np.random.randn(mx, d)
    y = np.random.randn(my, d)
    K = sklkernels.RBF(1.0)(x, y)
    scale = np.ones((n))
    ell = np.ones((n))
    kernel = QuadExp(n, Euclid.distance, scale=scale, ell=ell)
    x_ = torch.tensor(x).transpose(-1, -2)
    y_ = torch.tensor(y).transpose(-1, -2)
    K_ = kernel(x_, y_)[0]
    assert np.allclose(K_.data.cpu().numpy(), K)


def test_matern_kernel():
    n = 1
    d = 3
    mx = 5
    my = 4
    x = np.random.randn(mx, d)
    y = np.random.randn(my, d)
    x_ = torch.tensor(x).transpose(-1, -2)
    y_ = torch.tensor(y).transpose(-1, -2)
    scale = np.ones((n))
    ell = np.ones((n))
    for nu in [0.5, 1.5, 2.5]:
        K = sklkernels.Matern(length_scale=1.0, nu=nu)(x, y)
        kernel = Matern(n, Euclid.distance, scale=scale, ell=ell, nu=nu)
        K_ = kernel(x_, y_)[0].data.cpu().numpy()
        assert np.allclose(K_, K)


def test_linear_kernel():
    n = 1
    d = 3
    mx = 5
    my = 4
    x = np.random.randn(mx, d)
    y = np.random.randn(my, d)
    x_ = torch.tensor(x).transpose(-1, -2)
    y_ = torch.tensor(y).transpose(-1, -2)
    scale = np.ones((n))
    K = sklkernels.DotProduct(sigma_0=0)(x, y)
    kernel = Linear(n, Euclid.distance, scale=scale)
    K_ = kernel(x_, y_)[0].data.cpu().numpy()
    assert np.allclose(K_, K)


if __name__ == '__main__':
    test_quad_exp_kernel()
    test_matern_kernel()
    test_linear_kernel()
    test_quad_exp_hyp_prms_dims()
    test_quad_exp_trK()
    test_kernels_diagK()
    test_kernels_run()
    print('Tested kernels')
