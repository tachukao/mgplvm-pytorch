import numpy as np
import torch
from torch import optim
import mgplvm as mgp
import matplotlib.pyplot as plt
import scipy.stats

torch.manual_seed(1)
np.random.seed(0)

torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


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
        lat_dist = mgp.ReLie(manif, m, n_samples, diagonal=False)
        kernel = mgp.kernels.QuadExp(n, manif.distance)
        lik = mgp.likelihoods.Gaussian(n)
        prior = mgp.priors.Uniform(manif)
        z = manif.inducing_points(n, n_z)
        mod = mgp.SVGPLVM(n,
                                 m,
                                 n_samples,
                                 z,
                                 kernel,
                                 lik,
                                 lat_dist,
                                 prior,
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
        assert err < 1E-3


def test_batch_training():
    """test that optimization with and without batching work
    cant check that they give the exact same result since the random draws will differ"""
    d = 1  # dims of latent space
    n = 8  # number of neurons
    m = 30  # number of conditions / time points
    n_z = 5  # number of inducing points
    n_samples = 2  # number of samples
    gen = mgp.syndata.Gen(mgp.syndata.Euclid(d),
                          n,
                          m,
                          variability=0.25,
                          n_samples=n_samples)
    Y = gen.gen_data()
    data = torch.tensor(Y).to(device)
    mods = []
    m0s = []
    m1s = []
    for batch_size in [None, 5]:
        torch.manual_seed(0)
        np.random.seed(0)
        # specify manifold, kernel and rdist
        manif = mgp.manifolds.Euclid(m, d)
        lat_dist = mgp.ReLie(manif,
                             m,
                             n_samples,
                             diagonal=True,
                             initialization='fa',
                             Y=Y,
                             sigma=0.01)
        kernel = mgp.kernels.QuadExp(n, manif.distance)
        lik = mgp.likelihoods.Gaussian(n)
        prior = mgp.priors.Uniform(manif)
        z = manif.inducing_points(n, n_z)
        mods.append(
            mgp.SVGPLVM(n,
                               m,
                               n_samples,
                               z,
                               kernel,
                               lik,
                               lat_dist,
                               prior,
                               whiten=True,
                               tied_samples=False).to(device))
        m0s.append(mods[-1].lat_dist.prms[0].detach().cpu().numpy().flatten())
        print(m0s[-1][:10])
        train_ps = mgp.crossval.training_params(max_steps=21,
                                                n_mc=16,
                                                burnin=1,
                                                lrate=1.5e-1,
                                                batch_size=batch_size,
                                                accumulate_gradient=True,
                                                print_every=21)
        _ = mgp.crossval.train_model(mods[-1], data, train_ps)
        print('')
        m1s.append(mods[-1].lat_dist.prms[0].detach().cpu().numpy().flatten())
        print(m1s[-1][:10])

    r = scipy.stats.pearsonr(m1s[0], m1s[1])[0]
    print('correlation:', r)
    ##check that all mean parameters have been updated similarly in both cases
    assert r > 0.95


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
        lat_dist = mgp.ReLie(manif, m, n_samples, diagonal=False)
        kernel = mgp.kernels.QuadExp(n, manif.distance)
        lik = mgp.likelihoods.Gaussian(n)
        z = manif.inducing_points(n, n_z)
        svgp = mgp.SVGP(kernel,
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
        assert err < 1e-3


if __name__ == '__main__':
    test_svgp_batching()
    test_svgplvm_batching()
    test_batch_training()
