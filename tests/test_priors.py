import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
import mgplvm as mgp

torch.set_default_dtype(torch.float64)
device = mgp.utils.get_device()


def test_GP_prior():
    device = mgp.utils.get_device("cuda")  # get_device("cpu")
    d = 3  # dims of latent space
    d2 = 2  # dims of ts
    n = 50  # number of neurons
    m = 150  # number of conditions / time points
    n_z = 15  # number of inducing points
    n_samples = 2  # number of samples
    l = 0.55 * np.sqrt(d)
    gen = mgp.syndata.Gen(mgp.syndata.Euclid(d),
                          n,
                          m,
                          variability=0.15,
                          l=l,
                          sigma=0.8,
                          beta=0.1,
                          n_samples=n_samples)
    sig0 = 1.5
    Y = gen.gen_data(ell=25, sig=1)
    data = torch.tensor(Y, device=device, dtype=torch.get_default_dtype())
    # specify manifold, kernel and rdist
    manif = mgp.manifolds.Euclid(m, d)
    scale = np.mean(np.std(Y, axis=-1), axis=0)
    sigma = np.mean(np.std(Y, axis=-1), axis=0)  # initialize noise
    kernel = mgp.kernels.QuadExp(n, manif.distance, scale=scale)

    #lat_dist = mgp.MVN(m, d, sigma=sig0)
    lat_dist = mgp.ReLie(manif,
                               m,
                               n_samples,
                               sigma=sig0,
                               initialization='random',
                               Y=Y)

    ###construct prior
    prior_manif = mgp.manifolds.Euclid(m, d2)
    prior_kernel = mgp.kernels.QuadExp(d,
                                        prior_manif.distance,
                                        learn_scale=False)
    ts = torch.arange(m, device=device,
                      dtype=torch.get_default_dtype())[None, None, :].repeat(
                          n_samples, d2, 1)
    prior = mgp.priors.GP(d,
                            m,
                            n_samples,
                            prior_manif,
                            prior_kernel,
                            n_z=20,
                            ts=ts,
                            d=d2)
    #prior = priors.Gaussian(manif)

    # generate model
    likelihood = mgp.likelihoods.Gaussian(n, sigma=torch.Tensor(sigma))
    z = manif.inducing_points(n, n_z)
    mod = mgp.models.SvgpLvm(n, m, n_samples, z, kernel, likelihood, lat_dist,
                             prior).to(device)

    ### test that training runs ###
    n_mc = 64

    mgp.optimisers.svgp.fit(data,
                            mod,
                            optimizer=optim.Adam,
                            n_mc=n_mc,
                            max_steps=5,
                            burnin=1,
                            lrate=10E-2,
                            print_every=50)

    # check that we are indeed optimizing different q_mu in the GP prior for each sample
    assert (not (torch.allclose(mod.prior.svgp.q_mu[0].detach().data,
                                mod.prior.svgp.q_mu[1].detach().data)))

    ### test that two ways of computing the prior agree ###
    g, lq = mod.lat_dist.sample(torch.Size([n_mc]), data, None)
    g = g.transpose(-1, -2)

    def for_batch(i):
        svgp_lik, svgp_kl = mod.prior.svgp.elbo(g[i:i + 1], ts)
        elbo = svgp_lik - svgp_kl
        return elbo

    ##### naive computation ####
    LLs1 = [for_batch(i) for i in range(n_mc)]
    elbo1_b = torch.stack([LL.sum() for LL in LLs1], dim=0)

    ##### try to batch things ####
    lik, kl = mod.prior.svgp.elbo(g, ts)
    elbo2_b = (lik - kl).sum(-1)

    #### print comparison ###
    print('ELBOs:', elbo1_b.sum().detach().data, elbo2_b.sum().detach().data)
    assert all(torch.isclose(elbo1_b.detach().data, elbo2_b.detach().data))

    print(elbo1_b[:2])
    print(elbo2_b[:2])


def test_ARP_runs():
    m, d, n, n_z, p = 10, 3, 5, 5, 1
    n_samples = 2
    Y = np.random.normal(0, 1, (n_samples, n, m))
    data = torch.tensor(Y, device=device, dtype=torch.get_default_dtype())
    for i, manif_type in enumerate(
        [mgp.manifolds.Euclid, mgp.manifolds.Torus, mgp.manifolds.So3]):
        manif = manif_type(m, d)
        print(manif.name)
        lat_dist = mgp.ReLie(manif,
                                   m,
                                   n_samples,
                                   sigma=0.4,
                                   diagonal=(True if i in [0, 1] else False))
        kernel = mgp.kernels.QuadExp(n, manif.distance, Y=Y)
        # generate model
        lik = mgp.likelihoods.Gaussian(n)
        prior = mgp.priors.ARP(p,
                                 manif,
                                 diagonal=(True if i in [0, 1] else False))
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

        # train model
        mgp.optimisers.svgp.fit(data,
                                mod,
                                max_steps=5,
                                n_mc=64,
                                optimizer=optim.Adam,
                                print_every=1000)


def fio_id(x):
    return x


def fio_ReLU(x):
    return torch.max(0, x)


def fio_tanh(x):
    return torch.tanh(x)


def test_LDS_prior_runs():
    m, d, n, n_z, p = 10, 3, 5, 5, 1
    n_samples = 2
    Y = np.random.normal(0, 1, (n_samples, n, m))
    data = torch.tensor(Y, device=device, dtype=torch.get_default_dtype())
    for i, fio in enumerate([fio_id, fio_ReLU, fio_tanh]):
        print('fio', i)
        manif = mgp.manifolds.Euclid(m, d)
        lat_dist = mgp.ReLie(manif,
                                   m,
                                   n_samples,
                                   sigma=0.4,
                                   diagonal=True)
        kernel = mgp.kernels.QuadExp(n, manif.distance, Y=Y)
        # generate model
        lik = mgp.likelihoods.Gaussian(n)
        prior = mgp.priors.DS(manif, fio=fio)
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

        # train model
        mgp.optimisers.svgp.fit(data,
                                mod,
                                max_steps=10,
                                n_mc=16,
                                optimizer=optim.Adam,
                                print_every=1000)

        A, Q = mod.prior.prms
        A = A.detach().cpu().numpy()
        eigs = np.linalg.eigvals(A)
        print(eigs, np.amax(eigs))
        assert np.amax(eigs) < (
            1 + 1e-10)  #assert stability allowing for numerical errors

    return


if __name__ == '__main__':
    #test_GP_prior()
    test_ARP_runs()
    test_LDS_prior_runs()
    print('Tested priors')
