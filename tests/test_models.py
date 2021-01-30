import numpy as np
import torch
from torch import optim
import mgplvm as mgp
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def test_svgplvm_runs():
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
    lat_dist = mgp.rdist.ReLie(manif, m, n_samples, diagonal=False)
    kernel = mgp.kernels.QuadExp(n, manif.distance)
    lik = mgp.likelihoods.Gaussian(n)
    lprior = mgp.lpriors.Uniform(manif)
    z = manif.inducing_points(n, n_z)
    mod = mgp.models.SvgpLvm(n, z, kernel, lik, lat_dist, lprior,
                             whiten=True).to(device)

    # train model
    mgp.optimisers.svgp.fit(Y,
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
    elbo = (svgp_elbo - kl).data.cpu().numpy() / np.prod(Y.shape)

    assert elbo < LL

    #### test that batching with svgplvm runs ####
    trained_model = mgp.optimisers.svgp.fit(Y,
                                            mod,
                                            device,
                                            optimizer=optim.Adam,
                                            max_steps=5,
                                            n_mc=64,
                                            batch_size=int(np.round(m / 2, 0)))


def test_svgp_batching():
    """
    test that batching with svgp gives an unbiased estimate of  the true elbo
    """
    d = 1  # dims of latent space
    n = 8  # number of neurons
    m = 100  # number of conditions / time points
    n_z = 10  # number of inducing points
    n_samples = 1  # number of samples
    gen = mgp.syndata.Gen(mgp.syndata.Euclid(d),
                          n,
                          m,
                          variability=0.25,
                          n_samples=n_samples)
    Y = gen.gen_data()
    manif = mgp.manifolds.Euclid(m, d)
    lat_dist = mgp.rdist.ReLie(manif, m, n_samples, diagonal=False)
    kernel = mgp.kernels.QuadExp(n, manif.distance)
    lik = mgp.likelihoods.Gaussian(n)
    z = manif.inducing_points(n, n_z)
    svgp = mgp.models.svgp.Svgp(kernel,
                                n,
                                z,
                                lik,
                                n_samples=n_samples,
                                whiten=True)
    mod = svgp.to(device)
    lat_dist = lat_dist.to(device)
    data = torch.tensor(Y).to(device)
    g = lat_dist.lat_gmu(data).transpose(-1, -2)

    # not batched
    svgp_lik, svgp_kl = mod.elbo(data, g)
    elbo = (svgp_lik - svgp_kl).sum().item()

    batch_size = 20

    def for_batch():
        batch_idxs = np.random.choice(m, batch_size, replace=False)
        y = data[..., batch_idxs]
        g = lat_dist.lat_gmu(data, batch_idxs=batch_idxs).transpose(-1, -2)
        svgp_lik, svgp_kl = mod.elbo(y, g)
        elbo = ((m / batch_size) * svgp_lik) - svgp_kl
        return elbo.sum().item()

    estimated_elbo = np.mean([for_batch() for _ in range(2000)])

    assert (np.abs((elbo - estimated_elbo) / (elbo + estimated_elbo)) < 1e-4)


if __name__ == '__main__':
    test_svgplvm_runs()
    test_svgp_batching()
