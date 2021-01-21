import mgplvm
from mgplvm import manifolds, rdist, kernels, likelihoods, lpriors, models, optimisers
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


def test_manifs_runs():
    m, d, n, n_z = 10, 3, 5, 5
    Y = np.random.normal(0, 1, (n, m))
    for i, manif_type in enumerate(
        [manifolds.Torus, manifolds.So3, manifolds.S3]):
        manif = manif_type(m, d)
        print(manif.name)
        lat_dist = mgplvm.rdist.ReLie(manif,
                                      m,
                                      sigma=0.4,
                                      diagonal=(True if i == 0 else False))
        kernel = mgplvm.kernels.QuadExp(n, manif.distance, Y=Y)
        # generate model
        lik = mgplvm.likelihoods.Gaussian(n)
        lprior = mgplvm.lpriors.Uniform(manif)
        z = manif.inducing_points(n, n_z)
        mod = mgplvm.models.SvgpLvm(n,
                                    z,
                                    kernel,
                                    lik,
                                    lat_dist,
                                    lprior,
                                    whiten=True).to(device)

        # train model
        trained_model = optimisers.svgp.fit(Y,
                                            mod,
                                            device,
                                            max_steps=5,
                                            n_mc=64,
                                            optimizer=optim.Adam,
                                            print_every=1000)


if __name__ == '__main__':
    test_euclid_dimensions()
    test_torus_dimensions()
    test_so3_dimensions()
    test_manifs_runs()
