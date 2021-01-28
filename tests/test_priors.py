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
    # specify manifold, kernel and rdist
    manif = mgp.manifolds.Euclid(m, d)
    alpha = np.mean(np.std(Y, axis=-1), axis=0)
    sigma = np.mean(np.std(Y, axis=-1), axis=0)  # initialize noise
    kernel = mgp.kernels.QuadExp(n, manif.distance, alpha=alpha)

    #lat_dist = mgp.rdist.MVN(m, d, sigma=sig0)
    lat_dist = mgp.rdist.ReLie(manif,
                               m,
                               n_samples,
                               sigma=sig0,
                               initialization='random',
                               Y=Y)

    ###construct prior
    lprior_manif = mgp.manifolds.Euclid(m, d2)
    lprior_kernel = mgp.kernels.QuadExp(d,
                                        lprior_manif.distance,
                                        learn_alpha=False)
    ts = torch.arange(m).to(device)[None, ...]
    lprior = mgp.lpriors.GP(d,
                            n_samples,
                            lprior_manif,
                            lprior_kernel,
                            n_z=20,
                            ts=ts.repeat(n_samples, d2),
                            tmax=m)
    #lprior = lpriors.Gaussian(manif)

    # generate model
    likelihood = mgp.likelihoods.Gaussian(n, variance=np.square(sigma))
    z = manif.inducing_points(n, n_z)
    mod = mgp.models.SvgpLvm(n, z, kernel, likelihood, lat_dist,
                             lprior).to(device)

    ### test that training runs ###
    n_mc = 64
    mgp.optimisers.svgp.fit(Y,
                            mod,
                            device,
                            optimizer=optim.Adam,
                            n_mc=n_mc,
                            max_steps=5,
                            burnin=100,
                            lrate=10E-2,
                            print_every=50)

    ### test that two ways of computing the prior agree ###
    #data = torch.tensor(Y).to(device)
    #g, lq = mod.lat_dist.sample(torch.Size([n_mc]), data, None)

    ##input to prior

    #def elbo_for_batch(i):
    #    x = g[i:i + 1].transpose(-1, -2)
    #    lik, kl = mod.lprior.svgp.elbo(1,
    #                                   x,
    #                                   ts.repeat(n_samples, d2).reshape(
    #                                       1, n_samples, d2, -1),
    #                                   sum_samples=False)
    #    return (lik.sum(-2) - kl)

    ##### naive computation ####
    #LLs1 = [elbo_for_batch(i) for i in range(n_mc)]
    #elbo1_b = torch.stack([LL.sum() for LL in LLs1], dim=0)

    ##### try to batch things ####
    #ts = ts.repeat(n_samples * n_mc, d2).reshape(1, n_mc * n_samples, d2, -1)
    #lik, kl = mod.lprior.svgp.elbo(1,
    #                               g.transpose(-1, -2).reshape(-1, d, m),
    #                               ts,
    #                               sum_samples=False)
    #elbo2_b = (lik.reshape(n_mc, n_samples, d).sum(-2) - kl).sum(-1)
    #print(elbo1_b.shape, elbo2_b.shape)

    #### print comparison ###
    #print('ELBOs:', elbo1_b.sum().detach().data, elbo2_b.sum().detach().data)
    #assert all(torch.isclose(elbo1_b.detach().data, elbo2_b.detach().data))

    #print(elbo1_b[:5])
    #print(elbo2_b[:5])


def test_ARP_runs():
    m, d, n, n_z, p = 10, 3, 5, 5, 1
    n_samples = 2
    Y = np.random.normal(0, 1, (n_samples, n, m))
    for i, manif_type in enumerate(
        [mgp.manifolds.Euclid, mgp.manifolds.Torus, mgp.manifolds.So3]):
        manif = manif_type(m, d)
        print(manif.name)
        lat_dist = mgp.rdist.ReLie(manif,
                                   m,
                                   n_samples,
                                   sigma=0.4,
                                   diagonal=(True if i in [0, 1] else False))
        kernel = mgp.kernels.QuadExp(n, manif.distance, Y=Y)
        # generate model
        lik = mgp.likelihoods.Gaussian(n)
        lprior = mgp.lpriors.ARP(p,
                                 manif,
                                 diagonal=(True if i in [0, 1] else False))
        z = manif.inducing_points(n, n_z)
        mod = mgp.models.SvgpLvm(n,
                                 z,
                                 kernel,
                                 lik,
                                 lat_dist,
                                 lprior,
                                 whiten=True).to(device)

        # train model
        mgp.optimisers.svgp.fit(Y,
                                mod,
                                device,
                                max_steps=5,
                                n_mc=64,
                                optimizer=optim.Adam,
                                print_every=1000)


if __name__ == '__main__':
    test_GP_prior()
    test_ARP_runs()
    print('Tested priors')
