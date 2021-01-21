import numpy as np
import torch
from torch import optim
import mgplvm
from mgplvm import kernels, rdist, models, optimisers, syndata, likelihoods
from mgplvm.manifolds import Torus, Euclid, So3
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def test_svgp_runs():
    """
    test that svgp runs without explicit check for correctness
    also test that burda log likelihood runs and is smaller than elbo
    """
    d = 1  # dims of latent space
    n = 5  # number of neurons
    m = 10  # number of conditions / time points
    n_z = 5  # number of inducing points
    n_samples = 1  # number of samples
    gen = syndata.Gen(syndata.Euclid(d), n, m, variability=0.25)
    sig0 = 1.5
    l = 0.4
    gen.set_param('l', l)
    Y = gen.gen_data()[0]
    Y = Y + np.random.normal(size=Y.shape) * np.mean(Y) / 3
    # specify manifold, kernel and rdist
    manif = Euclid(m, d)
    lat_dist = mgplvm.rdist.ReLie(manif, m, sigma=sig0, diagonal=False)
    # initialize signal variance
    alpha = np.std(Y, axis=1)
    kernel = kernels.QuadExp(n, manif.distance, alpha=alpha)
    # generate model
    sigma = np.std(Y, axis=1)  # initialize noise
    lik = likelihoods.Gaussian(n, variance=np.square(sigma))
    lprior = mgplvm.lpriors.Uniform(manif)
    z = manif.inducing_points(n, n_z)
    mod = models.SvgpLvm(n, z, kernel, lik, lat_dist, lprior,
                         whiten=True).to(device)

    # train model
    trained_model = optimisers.svgp.fit(Y,
                                        mod,
                                        device,
                                        optimizer=optim.Adam,
                                        max_steps=5,
                                        burnin=5 / 2E-2,
                                        n_mc=64,
                                        lrate=2E-2,
                                        print_every=1000)

    ### test burda log likelihood ###
    LL = mod.calc_LL(torch.tensor(Y).to(device), 128)
    svgp_elbo, kl = mod.forward(torch.tensor(Y).to(device), 128)
    elbo = (svgp_elbo - kl) / (Y.shape[0] * Y.shape[1])

    assert elbo < LL


if __name__ == '__main__':
    test_svgp_runs()
