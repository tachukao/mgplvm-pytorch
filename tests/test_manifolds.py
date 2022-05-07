import mgplvm
from mgplvm import manifolds, kernels, likelihoods, priors, models, optimisers
import numpy as np
import torch
from torch import optim

torch.set_default_dtype(torch.float64)
device = mgplvm.utils.get_device()


def test_euclid_dimensions():
    e3 = manifolds.Euclid(10, 3)
    assert e3.d == 3
    assert e3.m == 10


def test_torus_dimensions():
    t3 = manifolds.Torus(10, 3)
    assert t3.d == 3
    assert t3.m == 10


def test_so3_dimensions():
    so3 = manifolds.So3(10)
    assert so3.d == 3
    assert so3.m == 10


def test_s3_dimensions():
    s3 = manifolds.S3(10)
    assert s3.d == 3
    assert s3.m == 10


def test_euclid_distance():
    n_mc = 7
    n_samples = 2
    n = 5
    mx = 8
    my = 6
    d = 3
    #x = torch.randn(2, 5, 1, d, mx)
    #y = torch.randn(2, 1, 4, d, my)
    x = torch.randn(n_mc, n_samples, n, d, mx)
    y = torch.randn(n_mc, n_samples, n, d, my)

    ell0 = torch.ones(n, d, 1) + torch.randn(n, d, 1) * 0.1
    ell1 = torch.ones(1, d, 1) + torch.randn(1, d, 1) * 0.1
    ell2 = torch.ones(n, 1, 1) + torch.randn(n, 1, 1) * 0.1
    ell3 = None

    manif = manifolds.Euclid(10, d)
    for ell in [ell0, ell1, ell2, ell3]:
        if ell is None:
            slow_dist = torch.square(x[..., None] - y[..., None, :]).sum(-3)
        else:
            slow_dist = (torch.square(x[..., None] - y[..., None, :]) /
                         ell[..., None]**2).sum(-3)
        slow_dist.clamp_min(0)
        dist = manif.distance(x, y, ell=ell)
        assert torch.allclose(slow_dist, dist)


def test_torus_distance():
    n_mc = 7
    n_samples = 2
    n = 5
    mx = 8
    my = 6
    d = 3
    #x = torch.randn(2, 5, 1, d, mx)
    #y = torch.randn(2, 1, 4, d, my)
    x = torch.randn(n_mc, n_samples, n, d, mx)
    y = torch.randn(n_mc, n_samples, n, d, my)

    ell0 = torch.ones(n, d, 1) + torch.randn(n, d, 1) * 0.1
    ell1 = torch.ones(1, d, 1) + torch.randn(1, d, 1) * 0.1
    ell2 = torch.ones(n, 1, 1) + torch.randn(n, 1, 1) * 0.1
    ell3 = None

    manif = manifolds.Torus(10, d)
    for ell in [ell0, ell1, ell2, ell3]:
        if ell is None:
            slow_dist = (2 -
                         2 * torch.cos(x[..., None] - y[..., None, :])).sum(-3)
        else:
            slow_dist = ((2 - 2 * torch.cos(x[..., None] - y[..., None, :])) /
                         ell[..., None]**2).sum(-3)
        slow_dist.clamp_min(0)
        dist = manif.distance(x, y, ell=ell)
        assert torch.allclose(slow_dist, dist)


def test_so3_distance():
    mx = 8
    my = 6
    d = 4
    x = torch.randn(2, 5, 1, d, mx)
    x = x / (1e-20 + x.square().sum(-2, keepdim=True).sqrt())
    y = torch.randn(2, 1, 4, d, my)
    y = y / (1e-20 + y.square().sum(-2, keepdim=True).sqrt())
    manif = manifolds.So3(10)
    z = (x[..., None] * y[..., None, :]).sum(-3)
    slow_dist = 4 * (1 - z.square())
    slow_dist.clamp_min_(0)
    dist = manif.distance(x, y)
    assert torch.allclose(slow_dist, dist)


def test_s3_distance():
    mx = 8
    my = 6
    d = 4
    x = torch.randn(2, 5, 1, d, mx)
    x = x / (1e-20 + x.square().sum(-2, keepdim=True).sqrt())
    y = torch.randn(2, 1, 4, d, my)
    y = y / (1e-20 + y.square().sum(-2, keepdim=True).sqrt())
    manif = manifolds.S3(10)
    z = (x[..., None] * y[..., None, :]).sum(-3)
    slow_dist = 2 * (1 - z)
    slow_dist.clamp_min_(0)
    dist = manif.distance(x, y)
    assert torch.allclose(slow_dist, dist)


def test_manifs_runs():
    m, d, n, n_z, n_samples = 10, 3, 5, 5, 2
    Y = np.random.normal(0, 1, (n_samples, n, m))
    data = torch.tensor(Y, dtype=torch.get_default_dtype(), device=device)
    for i, manif_type in enumerate(
        [manifolds.Torus, manifolds.So3, manifolds.S3]):
        manif = manif_type(m, d)
        print(manif.name)
        lat_dist = mgplvm.ReLie(manif,
                                m,
                                n_samples,
                                sigma=0.4,
                                diagonal=(True if i == 0 else False))
        kernel = mgplvm.kernels.QuadExp(n, manif.distance, Y=Y)
        # generate model
        lik = mgplvm.likelihoods.Gaussian(n)
        prior = mgplvm.priors.Uniform(manif)
        z = manif.inducing_points(n, n_z)
        mod = mgplvm.models.SvgpLvm(n,
                                    m,
                                    n_samples,
                                    z,
                                    kernel,
                                    lik,
                                    lat_dist,
                                    prior,
                                    whiten=True).to(device)

        # train model
        trained_model = optimisers.svgp.fit(data,
                                            mod,
                                            max_steps=5,
                                            n_mc=64,
                                            optimizer=optim.Adam,
                                            print_every=1000)


if __name__ == '__main__':
    test_euclid_dimensions()
    test_torus_dimensions()
    test_so3_dimensions()
    test_euclid_distance()
    test_torus_distance()
    test_so3_distance()
    test_s3_distance()
    test_manifs_runs()
