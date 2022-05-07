import numpy as np
import torch
from torch import optim
import mgplvm as mgp
import matplotlib.pyplot as plt

torch.manual_seed(1)
np.random.seed(0)

torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def test_svgplvm_LL():
    """
    test that svgplvm runs without explicit check for correctness
    also test that burda log likelihood runs and is smaller than elbo
    """
    d = 1  # dims of latent space
    n = 8  # number of neurons
    m = 10  # number of conditions / time points
    n_z = 5  # number of inducing points
    n_samples = 2  # number of samples
    gen = mgp.syndata.Gen(mgp.syndata.Euclid(d),
                          n,
                          m,
                          variability=0.25,
                          n_samples=n_samples)
    Y = gen.gen_data()
    # specify manifold, kernel and rdist
    manif = mgp.manifolds.Euclid(m, d)
    lat_dist = mgp.ReLie(manif, m, n_samples, diagonal=False)
    kernel = mgp.kernels.QuadExp(n, manif.distance)
    lik = mgp.likelihoods.Gaussian(n)
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

    data = torch.tensor(Y, device=device, dtype=torch.get_default_dtype())
    # train model
    mgp.optimisers.svgp.fit(data,
                            mod,
                            optimizer=optim.Adam,
                            max_steps=5,
                            burnin=5 / 2E-2,
                            n_mc=64,
                            lrate=2E-2,
                            print_every=1000)

    ### test burda log likelihood ###
    LL = mod.calc_LL(data, 128)
    svgp_elbo, kl = mod.forward(data, 128)
    elbo = (svgp_elbo - kl).data.cpu().numpy() / np.prod(Y.shape)

    assert elbo < LL


def test_lgplvm_LL():
    """
    test that Lgplvm and Lvgplvm run without explicit check for correctness
    also test that burda log likelihood runs and is smaller than elbo
    """
    d = 3  # dims of latent space
    n = 8  # number of neurons
    m = 10  # number of conditions / time points
    n_z = 5  # number of inducing points
    n_samples = 2  # number of samples
    gen = mgp.syndata.Gen(mgp.syndata.Euclid(d),
                          n,
                          m,
                          variability=0.25,
                          n_samples=n_samples)
    Y = gen.gen_data()
    # specify manifold, kernel and rdist
    for nmod in range(3):
        manif = mgp.manifolds.Euclid(m, d)
        lat_dist = mgp.ReLie(manif, m, n_samples, diagonal=False)
        kernel = mgp.kernels.QuadExp(n, manif.distance)
        lik = mgp.likelihoods.Gaussian(n)
        prior = mgp.priors.Uniform(manif)
        z = manif.inducing_points(n, n_z)
        if nmod in [0, 1]:
            mod = mgp.models.Lgplvm(n,
                                    m,
                                    d,
                                    n_samples,
                                    lat_dist,
                                    prior,
                                    Bayesian=(nmod == 1),
                                    Y=Y).to(device)
        else:
            mod = mgp.models.Lvgplvm(n, m, d, n_samples, lat_dist, prior,
                                     lik).to(device)

        data = torch.tensor(Y, device=device, dtype=torch.get_default_dtype())
        # train model
        mgp.optimisers.svgp.fit(data,
                                mod,
                                optimizer=optim.Adam,
                                max_steps=5,
                                burnin=5 / 2E-2,
                                n_mc=64,
                                lrate=2E-2,
                                print_every=1000)

        ### test burda log likelihood ###
        LL = mod.calc_LL(data, 128)
        svgp_elbo, kl = mod.forward(data, 128)
        elbo = (svgp_elbo - kl).data.cpu().numpy() / np.prod(Y.shape)

        assert elbo < LL


if __name__ == '__main__':
    test_lgplvm_LL()
    test_svgplvm_LL()
