import mgplvm
from mgplvm import manifolds, rdist, kernels, likelihoods, lpriors, models, training
import numpy as np
from torch import optim

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
    
def test_so3_runs():
    m, d, n, n_z = 10, 3, 5, 5
    manif = manifolds.So3(m, d)
    lat_dist = mgplvm.rdist.ReLie(manif, m, sigma=0.4)
    Y = np.random.normal(0, 1, (n, m, 1))
    kernel = mgplvm.kernels.QuadExp(n, manif.distance, Y = Y)
    # generate model
    device = mgplvm.utils.get_device()
    lik = mgplvm.likelihoods.Gaussian(n)
    lprior = mgplvm.lpriors.Uniform(manif)
    z = manif.inducing_points(n, n_z)
    mod = mgplvm.models.SvgpLvm(n, z, kernel, lik, lat_dist, lprior,
                         whiten=True).to(device)

    # train model
    trained_model = mgplvm.training.svgp(Y,
                                  mod,
                                  device,
                                  optimizer=optim.Adam,
                                  print_every=1000)
    
if __name__ == '__main__':
    test_euclid_dimensions()
    test_torus_dimensions()
    test_so3_dimensions()
    test_so3_runs()
    print('Tested manifolds')
    