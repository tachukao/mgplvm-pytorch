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
    lat_dist = mgp.rdist.ReLie(manif, m, n_samples, diagonal=False)
    kernel = mgp.kernels.QuadExp(n, manif.distance)
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


def test_svgplvm_batching():
    """
    test that svgplvm runs without explicit check for correctness
    also test that burda log likelihood runs and is smaller than elbo
    """
    d = 1  # dims of latent space
    n = 8  # number of neurons
    m = 30  # number of conditions / time points
    n_z = 5  # number of inducing points
    n_samples = 30  # number of samples
    gen = mgp.syndata.Gen(mgp.syndata.Euclid(d),
                          n,
                          m,
                          variability=0.25,
                          n_samples=n_samples)
    for tied_samples in [True, False]:
        Y = gen.gen_data()
        # specify manifold, kernel and rdist
        manif = mgp.manifolds.Euclid(m, d)
        lat_dist = mgp.rdist.ReLie(manif, m, n_samples, diagonal=False)
        kernel = mgp.kernels.QuadExp(n, manif.distance)
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
                                 whiten=True,
                                 tied_samples=tied_samples).to(device)

        data = torch.tensor(Y).to(device)
        n_mc = 16
        svgp_elbo, kl = mod.forward(data, n_mc=n_mc)
        elbo = (svgp_elbo - kl).sum().item()

        batch_size = 10
        sample_size = 10

        def for_batch():
            batch_idxs = np.random.choice(m, batch_size, replace=False)
            sample_idxs = np.random.choice(n_samples,
                                           sample_size,
                                           replace=False)
            batch = data[sample_idxs][:, :, batch_idxs]
            svgp_elbo, svgp_kl = mod.forward(batch,
                                             n_mc=n_mc,
                                             batch_idxs=batch_idxs,
                                             sample_idxs=sample_idxs)
            elbo = svgp_elbo - svgp_kl
            return elbo.sum().item()

        est_elbos = [for_batch() for _ in range(500)]
        err = np.abs(elbo - np.mean(est_elbos)) / np.linalg.norm(est_elbos)
        assert err < 1E-4


def test_svgp_batching():
    """
    test that batching with svgp gives an unbiased estimate of  the true elbo
    """
    d = 1  # dims of latent space
    n = 8  # number of neurons
    m = 30  # number of conditions / time points
    n_z = 10  # number of inducing points
    n_samples = 30  # number of samples
    gen = mgp.syndata.Gen(mgp.syndata.Euclid(d),
                          n,
                          m,
                          variability=0.25,
                          n_samples=n_samples)
    for tied_samples in [True, False]:
        Y = gen.gen_data()
        manif = mgp.manifolds.Euclid(m, d)
        lat_dist = mgp.rdist.ReLie(manif, m, n_samples, diagonal=False)
        kernel = mgp.kernels.QuadExp(n, manif.distance)
        lik = mgp.likelihoods.Gaussian(n)
        z = manif.inducing_points(n, n_z)
        svgp = mgp.models.svgp.Svgp(kernel,
                                    n,
                                    m,
                                    n_samples,
                                    z,
                                    lik,
                                    whiten=True,
                                    tied_samples=tied_samples)
        mod = svgp.to(device)
        lat_dist = lat_dist.to(device)
        data = torch.tensor(Y).to(device)
        g = lat_dist.lat_gmu(data).transpose(-1, -2)

        # not batched
        svgp_lik, svgp_kl = mod.elbo(data, g)
        elbo = (svgp_lik - svgp_kl).sum().item()

        batch_size = 10
        sample_size = 10

        def for_batch():
            batch_idxs = np.random.choice(m, batch_size, replace=False)
            sample_idxs = np.random.choice(n_samples,
                                           sample_size,
                                           replace=False)
            y = data[sample_idxs][..., batch_idxs]
            g = lat_dist.lat_gmu(data,
                                 batch_idxs=batch_idxs,
                                 sample_idxs=sample_idxs).transpose(-1, -2)
            svgp_lik, svgp_kl = mod.elbo(y, g, sample_idxs=sample_idxs)
            elbo = svgp_lik - svgp_kl
            return elbo.sum().item()

        est_elbos = [for_batch() for _ in range(500)]
        err = np.abs(elbo - np.mean(est_elbos)) / np.linalg.norm(est_elbos)
        assert err < 1e-4


if __name__ == '__main__':
    test_svgp_batching()
    test_svgplvm_batching()
    test_svgplvm_LL()
