import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
import mgplvm
from mgplvm import kernels, rdist, models, training, syndata
from mgplvm.manifolds import Torus, Euclid
from mgplvm import lpriors
torch.set_default_dtype(torch.float64)


def test_GP_prior():
    device = mgplvm.utils.get_device("cuda")  # get_device("cpu")
    d = 1  # dims of latent space
    n = 100  # number of neurons
    m = 250  # number of conditions / time points
    n_z = 15  # number of inducing points
    n_samples = 1  # number of samples
    l = float(0.55 * np.sqrt(d))
    gen = syndata.Gen(syndata.Euclid(d),
                      n,
                      m,
                      variability=0.15,
                      l=l,
                      sigma=0.8,
                      beta=0.1)
    sig0 = 1.5
    Y = gen.gen_data(ell=25, sig=1)

    # specify manifold, kernel and rdist
    manif = Euclid(m, d, initialization='random', Y=Y[:, :, 0])

    #lat_dist = mgplvm.rdist.MVN(m, d, sigma=sig0)
    lat_dist = mgplvm.rdist.ReLie(manif, m, sigma=sig0)
    alpha = np.mean(np.std(Y, axis=1), axis=1)
    kernel = kernels.QuadExp(n, manif.distance, alpha=alpha)

    ###construct prior
    lprior_kernel = kernels.QuadExp(d, manif.distance, learn_alpha=False)
    lprior = lpriors.GP(manif, lprior_kernel, n_z=20, tmax=m)
    #lprior = lpriors.Gaussian(manif)

    # generate model
    sigma = np.mean(np.std(Y, axis=1), axis=1)  # initialize noise
    likelihood = mgplvm.likelihoods.Gaussian(n, variance=np.square(sigma))
    z = manif.inducing_points(n, n_z)
    mod = models.SvgpLvm(n, z, kernel, likelihood, lat_dist, lprior).to(device)

    ### test that training runs ###
    ts = torch.arange(m).to(device)
    n_mc = 64
    trained_mod = training.svgp(Y,
                                mod,
                                device,
                                optimizer=optim.Adam,
                                outdir='none',
                                max_steps=5,
                                burnin=100,
                                n_mc=n_mc,
                                lrate=10E-2,
                                print_every=50,
                                n_svgp=0,
                                ts=ts,
                                callback=None)

    ### test that two ways of computing the prior agree ###
    data = torch.tensor(Y).to(device)
    ts = torch.arange(m).to(device)
    _, _, n_samples = data.shape  #n x mx x n_samples
    g, lq = mod.lat_dist.sample(torch.Size([n_mc]), None)

    x = g  #input to prior

    #### naive computation ####
    LLs1 = [
        mod.lprior.svgp.elbo(1, x[i, :, :, None].permute(1, 0, 2),
                             ts.reshape(1, 1, -1)) for i in range(x.shape[0])
    ]
    lik1_b = torch.stack([LL[0] for LL in LLs1],
                       dim=0)  #sum likelihood across samples of x
    kl1_b = torch.stack([LL[1] for LL in LLs1],
                      dim=0)  #sum KL across samples of x
    lik1, kl1 = lik1_b.sum(), kl1_b.sum()

    #### try to batch things ####
    lik2_b, kl2_b = mod.lprior.svgp.elbo(x.shape[0], x.permute(2, 1, 0),
                                     ts.reshape(1, 1, -1), by_sample = True)
    lik2, kl2 = lik2_b.sum(), kl2_b.sum()
    print(lik1_b.shape, kl1_b.shape, lik2_b.shape, kl2_b.shape)
    

    ### print comparison ###
    print('ELBOs:', (lik1 - kl1).detach().data, (lik2 - kl2).detach().data)
    print('likelihoods:', lik1.detach().data, lik2.detach().data)
    print('KLs:', kl1.detach().data, kl2.detach().data)
    assert torch.isclose((lik1 - kl1).detach().data,
                         (lik2 - kl2).detach().data)
    
    
    print(kl1_b[:5])
    print(kl2_b[:5])
    print(lik1_b[:5])
    print(lik2_b[:5])


if __name__ == '__main__':
    test_GP_prior()
    print('done')
