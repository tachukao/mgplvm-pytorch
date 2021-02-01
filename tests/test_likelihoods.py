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


def test_likelihood_runs():
    """
    test that all likelihoods run without explicit check for correctness
    also check that burda LL is greater than elbo for all likelihoods
    """
    d = 1  # dims of latent space
    n = 5  # number of neurons
    m = 10  # number of conditions / time points
    n_z = 5  # number of inducing points
    n_samples = 2  # number of samples
    gen = syndata.Gen(syndata.Euclid(d),
                      n,
                      m,
                      variability=0.25,
                      n_samples=n_samples)
    Y = gen.gen_data()
    Y = np.round(Y - np.amin(Y))
    print(Y.shape)

    for lik in [
            likelihoods.Gaussian(n),
            likelihoods.Poisson(n),
            # using the Gauss Hermite
            likelihoods.Poisson(n, inv_link=lambda x: torch.exp(x + 2)),
            likelihoods.NegativeBinomial(n)
    ]:
        # specify manifold, kernel and rdist
        manif = Euclid(m, d)
        lat_dist = mgplvm.rdist.ReLie(manif, m, n_samples, diagonal=False)
        # initialize signal variance
        kernel = kernels.QuadExp(n, manif.distance)
        # generate model
        lprior = mgplvm.lpriors.Uniform(manif)
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

        # train model
        optimisers.svgp.fit(Y,
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
        print("once")
        svgp_elbo, kl = mod.forward(torch.tensor(Y).to(device), 128)
        elbo = (svgp_elbo - kl) / np.prod(Y.shape)

        assert elbo <= LL


if __name__ == '__main__':
    test_likelihood_runs()
    print('Tested likelihoods')
