import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
import mgplvm as mgp
torch.set_default_dtype(torch.float64)
device = mgp.utils.get_device()


def test_GP_lat_prior():
    device = mgp.utils.get_device("cuda")  # get_device("cpu")
    d = 3  # dims of latent space
    n = 50  # number of neurons
    m = 60  # number of conditions / time points
    n_z = 15  # number of inducing points
    n_samples = 10  # number of samples
    
    #generate from GPFA generative model
    ts = np.array([np.arange(m) for nsamp in range(n_samples)])[:, None, :]
    
    dts_sq = (ts[..., None] - ts[..., None, :])**2 #(n_samples x 1 x m x m)
    dts_sq = np.sum(dts_sq, axis = -3) #(n_samples x m x m)
    K = np.exp( - dts_sq / 20**2 ) +1e-6 * np.eye(m)[None, ...]
    L = np.linalg.cholesky(K)
    us = np.random.normal(0, 1, size = (m, d))
    xs = L @ us #(n_samples x m x d)
    print('xs:', xs.shape)
    w = np.random.normal(0, 1, size = (n, d))
    Y = w @ xs.transpose(0, 2, 1) #(n_samples x n x m)
    print('Y:', Y.shape)

    data = torch.tensor(Y, device=device, dtype=torch.get_default_dtype())
    # specify manifold, kernel and rdist
    manif = mgp.manifolds.Euclid(m, d)
    kernel = mgp.kernels.Linear(n, d)

    #latent distribution is itself a GP
    lat_dist = mgp.rdist.lat_GP(manif,
                                 m,
                                 n_samples,
                                 torch.Tensor(ts).to(device),
                                   Y = Y)

    ###construct prior
    lprior = mgp.lpriors.GP_full(d,
                                 m,
                                 n_samples,
                                 manif,
                                 torch.Tensor(ts).to(device))

    # generate model
    likelihood = mgp.likelihoods.Gaussian(n, Y = Y)
    z = manif.inducing_points(n, n_z)
    mod = mgp.models.SvgpLvm(n, m, n_samples, z, kernel, likelihood, lat_dist,
                             lprior).to(device)

    ### test that training runs ###
    n_mc = 16

    mgp.optimisers.svgp.fit(data,
                            mod,
                            optimizer=optim.Adam,
                            n_mc=n_mc,
                            max_steps=5,
                            burnin=1,
                            lrate=10E-2,
                            print_every=50)

    
if __name__ == '__main__':
    test_GP_lat_prior()
