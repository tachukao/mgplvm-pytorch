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


def test_linear_ard():
    """
    test that svgplvm runs without explicit check for correctness
    also test that burda log likelihood runs and is smaller than elbo
    """

    x = np

    d0 = 2  # dims of true latent space
    d = d0 + 2  # gplvm dim
    n = 15  # number of neurons
    m = 30  # number of conditions / time points
    n_z = 5  # number of inducing points
    n_samples = 1  # number of samples

    x = np.random.normal(0, 1, size=(n_samples, m, d0))  #generate latents
    C = np.random.normal(0, 1, size=(n, d0))  #actoor matrix
    Y = C @ x.transpose(0, 2, 1)
    assert Y.shape == (n_samples, n, m)

    sigs = np.random.uniform(0, 0.5, size=n)
    Y = Y + np.random.normal(0, np.tile(sigs[None, ..., None],
                                        (n_samples, 1, m)))  #add noise

    # specify manifold, kernel and rdist
    manif = mgp.manifolds.Euclid(m, d)  #over-parameterize
    lat_dist = mgp.rdist.ReLie(manif,
                               m,
                               n_samples,
                               diagonal=True,
                               sigma=0.2,
                               initialization='fa',
                               Y=Y)
    kernel = mgp.kernels.Linear(n, d, ard=True, learn_scale=False)
    lik = mgp.likelihoods.Gaussian(n)
    lprior = mgp.lpriors.Uniform(manif)
    z = manif.inducing_points(n, n_z)
    mod = mgp.models.SvgpLvm(n,
                             m,
                             n_samples,
                             z,
                             kernel,
                             lik,
                             lat_dist,
                             lprior,
                             whiten=True).to(device)

    trained_mod = mgp.optimisers.svgp.fit(torch.tensor(Y).to(device),
                                          mod,
                                          optimizer=optim.Adam,
                                          max_steps=300,
                                          burnin=30,
                                          n_mc=5,
                                          lrate=7.5E-2,
                                          print_every=50)

    ells = mod.kernel.ell
    ells = np.sort(ells.detach().cpu().numpy()**(-1))
    print('\n', ells)

    for i in range(
            d -
            d0):  #more than a standaard deeviaation away from the other ells
        assert ells[d0 + i] > (ells[d0 - 1] + np.std(ells[:d0]))


if __name__ == '__main__':
    test_linear_ard()
