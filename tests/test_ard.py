import numpy as np
import torch
from torch import optim
import mgplvm as mgp
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def gen_ard_data(d0=2, d=4, n=25, m=50, n_z=15, n_samples=1, lin=True):
    x = np.random.normal(0, 1, size=(n_samples, m, d0))  #generate latents
    C = np.random.normal(0, 1, size=(n, d0))  #actoor matrix

    if lin:
        Y = C @ x.transpose(0, 2, 1)
        sigs = np.random.uniform(0, 0.25, size=n)
    else:
        Y = np.sin(1 * C @ x.transpose(0, 2, 1))
        sigs = np.random.uniform(0, 0.25, size=n)

    assert Y.shape == (n_samples, n, m)

    Y = Y + np.random.normal(0, np.tile(sigs[None, ..., None],
                                        (n_samples, 1, m)))  #add noise

    return d0, d, n, m, n_z, n_samples, Y


def test_linear_ard():
    """
    test that the linear ARD functionality correctly discards the unwanted dimensions
    """

    d0, d, n, m, n_z, n_samples, Y = gen_ard_data()

    # specify manifold, kernel and rdist
    manif = mgp.manifolds.Euclid(m, d)  #over-parameterize
    lat_dist = mgp.ReLie(manif,
                         m,
                         n_samples,
                         diagonal=True,
                         sigma=0.3,
                         initialization='fa',
                         Y=Y)
    kernel = mgp.kernels.Linear(n, d, ard=True, learn_scale=False)
    lik = mgp.likelihoods.Gaussian(n, sigma=torch.ones(n) * 0.5)
    prior = mgp.priors.Uniform(manif)
    z = manif.inducing_points(n, n_z)
    mod = mgp.models.SvgpLvm(n,
                             m,
                             n_samples,
                             z,
                             kernel,
                             lik,
                             lat_dist,
                             prior,
                             whiten=True).to(device)

    trained_mod = mgp.optimisers.svgp.fit(torch.tensor(Y).to(device),
                                          mod,
                                          optimizer=optim.Adam,
                                          max_steps=700,
                                          burnin=50,
                                          n_mc=5,
                                          lrate=20E-2,
                                          print_every=50)

    ells = (mod.svgp.kernel.input_scale)**(-1)
    ells = np.sort(ells.detach().cpu().numpy())
    print('\n', ells)

    for i in range(d - d0):
        #more than three standard deviations away from the other ells
        assert ells[d0 + i] > (ells[d0 - 1] + 3 * np.std(ells[:d0]))


def test_rbf_ard():
    """
    test that the RBF ARD functionality correctly discards the unwanted dimensions
    """

    d0, d, n, m, n_z, n_samples, Y = gen_ard_data(lin=False)

    # specify manifold, kernel and rdist
    manif = mgp.manifolds.Euclid(m, d)
    lat_dist = mgp.ReLie(manif,
                         m,
                         n_samples,
                         diagonal=True,
                         sigma=0.5,
                         initialization='fa',
                         Y=Y)
    kernel = mgp.kernels.QuadExp(n,
                                 manif.distance,
                                 Y=Y,
                                 d=d,
                                 ell_byneuron=False)
    print(d, kernel.ell.shape)
    lik = mgp.likelihoods.Gaussian(n, sigma=torch.ones(n) * 0.25)
    prior = mgp.priors.Uniform(manif)
    z = manif.inducing_points(n, n_z)
    mod = mgp.models.SvgpLvm(n,
                             m,
                             n_samples,
                             z,
                             kernel,
                             lik,
                             lat_dist,
                             prior,
                             whiten=True).to(device)

    trained_mod = mgp.optimisers.svgp.fit(torch.tensor(Y).to(device),
                                          mod,
                                          optimizer=optim.Adam,
                                          max_steps=700,
                                          burnin=50,
                                          n_mc=12,
                                          lrate=15E-2,
                                          print_every=50)

    ells = np.sort(mod.svgp.kernel.ell.detach().cpu().numpy()[0, :])
    print('\n', ells)

    for i in range(d - d0):
        #more than three standard deviations away from the other ells
        assert ells[d0 + i] > (ells[d0 - 1] + 2 * np.std(ells[:d0]))


if __name__ == '__main__':
    test_linear_ard()
    test_rbf_ard()
