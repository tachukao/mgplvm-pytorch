import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
import mgplvm
from mgplvm import manifolds, rdist, kernels, likelihoods, lpriors, models, optimisers
torch.set_default_dtype(torch.float64)
device = mgplvm.utils.get_device()


def test_GP_prior():
    device = mgplvm.utils.get_device("cuda")  # get_device("cpu")
    d = 1  # dims of latent space
    n = 100  # number of neurons
    m = 250  # number of conditions / time points
    n_z = 15  # number of inducing points
    n_samples = 1  # number of samples
    l = float(0.55 * np.sqrt(d))
    gen = mgplvm.syndata.Gen(mgplvm.syndata.Euclid(d),
                             n,
                             m,
                             variability=0.15,
                             l=l,
                             sigma=0.8,
                             beta=0.1)
    sig0 = 1.5
    Y = gen.gen_data(ell=25, sig=1)

    # specify manifold, kernel and rdist
    manif = mgplvm.manifolds.Euclid(m, d)

    #lat_dist = mgplvm.rdist.MVN(m, d, sigma=sig0)
    lat_dist = mgplvm.rdist.ReLie(manif,
                                  m,
                                  sigma=sig0,
                                  initialization='random',
                                  Y=Y[:, :, 0])
    alpha = np.mean(np.std(Y, axis=1), axis=1)
    kernel = mgplvm.kernels.QuadExp(n, manif.distance, alpha=alpha)

    ###construct prior
    lprior_kernel = mgplvm.kernels.QuadExp(d,
                                           manif.distance,
                                           learn_alpha=False)
    lprior = mgplvm.lpriors.GP(manif, lprior_kernel, n_z=20, tmax=m)
    #lprior = lpriors.Gaussian(manif)

    # generate model
    sigma = np.mean(np.std(Y, axis=1), axis=1)  # initialize noise
    likelihood = mgplvm.likelihoods.Gaussian(n, variance=np.square(sigma))
    z = manif.inducing_points(n, n_z)
    mod = mgplvm.models.SvgpLvm(n, z, kernel, likelihood, lat_dist,
                                lprior).to(device)

    ### test that training runs ###
    ts = torch.arange(m).to(device)
    n_mc = 64
    trained_mod = mgplvm.optimisers.svgp.fit(Y,
                                             mod,
                                             device,
                                             optimizer=optim.Adam,
                                             max_steps=5,
                                             burnin=100,
                                             n_mc=n_mc,
                                             lrate=10E-2,
                                             print_every=50,
                                             ts=ts)

    ### test that two ways of computing the prior agree ###
    data = torch.tensor(Y).to(device)
    ts = torch.arange(m).to(device)
    _, _, n_samples = data.shape  #n x mx x n_samples
    g, lq = mod.lat_dist.sample(torch.Size([n_mc]), data, None)

    x = g  #input to prior

    #### naive computation ####
    LLs1 = [
        mod.lprior.svgp.elbo(1, x[i, :, :, None].permute(1, 0, 2),
                             ts.reshape(1, 1, -1)) for i in range(x.shape[0])
    ]
    elbo1_b = torch.stack([LL.sum() for LL in LLs1], dim=0)

    #### try to batch things ####
    elbo2_b = mod.lprior.svgp.elbo(x.shape[0], x.permute(2, 1, 0),
                                   ts.reshape(1, 1, -1))
    elbo2_b = elbo2_b.sum(0).sum(0)
    print(elbo1_b.shape, elbo2_b.shape)

    ### print comparison ###
    print('ELBOs:', elbo1_b.sum().detach().data, elbo2_b.sum().detach().data)
    assert all(torch.isclose(elbo1_b.detach().data, elbo2_b.detach().data))

    print(elbo1_b[:5])
    print(elbo2_b[:5])

def test_ARP_runs():
    m, d, n, n_z, p = 10, 3, 5, 5, 1
    Y = np.random.normal(0, 1, (n, m, 1))
    for i, manif_type in enumerate(
        [manifolds.Euclid, manifolds.Torus, manifolds.So3]):
        manif = manif_type(m, d)
        print(manif.name)
        lat_dist = mgplvm.rdist.ReLie(manif,
                                      m,
                                      sigma=0.4,
                                      diagonal=(True if i in [0,1] else False))
        kernel = mgplvm.kernels.QuadExp(n, manif.distance, Y=Y)
        # generate model
        lik = mgplvm.likelihoods.Gaussian(n)
        lprior = mgplvm.lpriors.ARP(p, manif, diagonal=(True if i in [0,1] else False))
        z = manif.inducing_points(n, n_z)
        mod = mgplvm.models.SvgpLvm(n,
                                    z,
                                    kernel,
                                    lik,
                                    lat_dist,
                                    lprior,
                                    whiten=True).to(device)

        # train model
        trained_model = optimisers.svgp.fit(Y,
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
