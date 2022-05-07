import numpy as np
import torch
from torch import optim
import mgplvm
from mgplvm import kernels, models, optimisers, syndata, likelihoods
from mgplvm import lat_dist as lat_dist_lib
from mgplvm.manifolds import Torus, Euclid, So3
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def test_cv_runs():
    """
    test that svgp runs without explicit check for correctness
    also test that burda log likelihood runs and is smaller than elbo
    """
    d = 1  # dims of latent space
    n = 8  # number of neurons
    m = 10  # number of conditions / time points
    n_z = 6  # number of inducing points
    n_samples = 1  # number of samples
    gen = syndata.Gen(syndata.Euclid(d),
                      n,
                      m,
                      variability=0.25,
                      n_samples=n_samples)
    Y = gen.gen_data()
    # specify manifold, kernel and lat_dist
    manif = Euclid(m, d)
    lat_dist = lat_dist_lib.ReLie(manif, m, n_samples, diagonal=False)
    kernel = kernels.QuadExp(n, manif.distance)
    lik = likelihoods.Gaussian(n)
    prior = mgplvm.priors.Uniform(manif)
    z = manif.inducing_points(n, n_z)
    mod = models.SvgpLvm(n,
                         m,
                         n_samples,
                         z,
                         kernel,
                         lik,
                         lat_dist,
                         prior,
                         whiten=True).to(device)

    ### run cv ###
    train_ps = mgplvm.crossval.training_params(lrate=5e-2,
                                               burnin=20,
                                               batch_size=None,
                                               max_steps=10,
                                               n_mc=32)
    mod, split = mgplvm.crossval.train_cv(mod, Y, device, train_ps, test=False)
    mgplvm.crossval.test_cv(mod, split, device, Print=True)


if __name__ == '__main__':
    test_cv_runs()
