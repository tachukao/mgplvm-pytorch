import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
import mgplvm as mgp
from sklearn.cross_decomposition import CCA
torch.set_default_dtype(torch.float64)
device = mgp.utils.get_device()


def test_GP_lat_prior():
    device = mgp.utils.get_device("cuda")  # get_device("cpu")
    d = 1  # dims of latent space
    dfit = 2 #dimensions of fitted space
    n = 50  # number of neurons
    m = 80  # number of conditions / time points
    n_z = 15  # number of inducing points
    n_samples = 10  # number of samples
    
    #generate from GPFA generative model
    ts = np.array([np.arange(m) for nsamp in range(n_samples)])[:, None, :]
    
    dts_sq = (ts[..., None] - ts[..., None, :])**2 #(n_samples x 1 x m x m)
    dts_sq = np.sum(dts_sq, axis = -3) #(n_samples x m x m)
    K = np.exp( - dts_sq / 10**2 ) +1e-6 * np.eye(m)[None, ...]
    L = np.linalg.cholesky(K)
    us = np.random.normal(0, 1, size = (m, d))
    xs = L @ us #(n_samples x m x d)
    print('xs:', xs.shape)
    w = np.random.normal(0, 1, size = (n, d))
    Y = w @ xs.transpose(0, 2, 1) #(n_samples x n x m)
    #Y = Y + np.random.normal(0, 0.2, size = Y.shape)
    Y = np.random.poisson( 4/d*(Y - np.amin(Y)) )
    print('Y:', Y.shape, np.std(Y), np.quantile(Y, 0.99))

    data = torch.tensor(Y, device=device, dtype=torch.get_default_dtype())
    # specify manifold, kernel and rdist
    manif = mgp.manifolds.Euclid(m, dfit)
    #kernel = mgp.kernels.Linear(n, dfit)
    kernel = mgp.kernels.Linear(n, dfit, ard=True, learn_scale=False)

    #latent distribution is itself a GP
    lat_dist = mgp.rdist.lat_GP(manif,
                                 m,
                                 n_samples,
                                 torch.Tensor(ts).to(device),
                                   Y = Y,
                               initialization = 'fa')

    ###construct prior
    lprior = mgp.lpriors.GP_full(dfit,
                                 m,
                                 n_samples,
                                 manif,
                                 torch.Tensor(ts).to(device))

    # generate model
    #likelihood = mgp.likelihoods.Gaussian(n, Y = Y, d = dfit)
    likelihood = mgp.likelihoods.NegativeBinomial(n, Y = Y)
    z = manif.inducing_points(n, n_z)
    mod = mgp.models.SvgpLvm(n, m, n_samples, z, kernel, likelihood, lat_dist,
                             lprior).to(device)

    ### test that training runs ###
    n_mc = 16
        
    def cb(mod, i ,loss):
        if i % 5 == 0:
            print('')
        return
    
    mgp.optimisers.svgp.fit(data,
                            mod,
                            optimizer=optim.Adam,
                            n_mc=n_mc,
                            max_steps=3000,
                            burnin=50,
                            lrate=5e-2,
                            print_every=5,
                           stop = cb)

    
    print('lat_dist (ell, scale):', mod.lat_dist.f.ell.flatten(), mod.lat_dist.f.scale.flatten())
    print('prior:', mod.lprior.ell.flatten())
    print('kernel:', (mod.kernel.input_scale)**(-1))
    
    
    mus = mod.lat_dist.prms[0].detach().cpu().numpy()
    print('mus:', np.std(mus, axis = (0, 1)))

    ex = 0
    xs = xs[ex, ...] - np.mean(xs[ex, ...], axis = 0, keepdims = True)
    mus = mus[ex, ...] - np.mean(mus[ex, ...], axis = 0, keepdims = True)
    
    #align
    if d == 1:
        mus = mus[:, :1]
        
    cca = CCA(n_components = d)
    X_, Y_ = cca.fit_transform(mus.reshape(-1, d), xs.reshape(-1,d))
    mus = mus @ cca.coef_
    xs = xs/np.std(xs, axis = 0, keepdims = True)
    mus = mus/np.std(mus, axis = 0, keepdims = True)
    
    print(mus.shape)
    plt.figure()
    if d >= 2:
        plt.plot(mus[:, 0], mus[:, 1], 'k-')
        plt.plot(xs[:, 0], xs[:, 1], 'b-')
    else:
        plt.plot(ts[0, 0, :], mus[:, 0], 'k-')
        plt.plot(ts[0, 0, :], xs[:, 0], 'b-')
    plt.xlabel('lat dim 1')
    plt.ylabel('lat dim 2')
    plt.legend(['inferred', 'true'], frameon = False)
    plt.savefig('figures/test_gp.png', bbox_inches = 'tight')
    plt.close()
    
if __name__ == '__main__':
    test_GP_lat_prior()
