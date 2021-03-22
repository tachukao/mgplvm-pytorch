import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
import mgplvm as mgp
from sklearn.cross_decomposition import CCA

torch.set_default_dtype(torch.float64)
device = mgp.utils.get_device()


def test_fast_GP_lat_prior():
    device = mgp.utils.get_device("cuda")  # get_device("cpu")
    d = 2  # dims of latent space
    dfit = 2  #dimensions of fitted space
    n = 50  # number of neurons
    m = 80  # number of conditions / time points
    n_z = 15  # number of inducing points
    n_samples = 10  # number of samples
    Poisson = False

    #generate from GPFA generative model
    ts = np.array([np.arange(m) for nsamp in range(n_samples)])[:, None, :]

    dts_sq = (ts[..., None] - ts[..., None, :])**2  #(n_samples x 1 x m x m)
    dts_sq = np.sum(dts_sq, axis=-3)  #(n_samples x m x m)
    K = np.exp(-dts_sq / (2 * 7**2)) + 1e-6 * np.eye(m)[None, ...]
    L = np.linalg.cholesky(K)
    us = np.random.normal(0, 1, size=(m, d))
    xs = L @ us  #(n_samples x m x d)
    print('xs:', xs.shape)
    w = np.random.normal(0, 1, size=(n, d))
    Y = w @ xs.transpose(0, 2, 1)  #(n_samples x n x m)
    if Poisson:
        Y = np.random.poisson(2 * (Y - np.amin(Y)))
    else:
        Y = Y + np.random.normal(0, 0.2, size=Y.shape)
    print('Y:', Y.shape, np.std(Y), np.quantile(Y, 0.99))

    data = torch.tensor(Y, device=device, dtype=torch.get_default_dtype())
    # specify manifold, kernel and rdist
    manif = mgp.manifolds.Euclid(m, dfit)
    #kernel = mgp.kernels.Linear(n, dfit)
    kernel = mgp.kernels.Linear(n,
                                dfit,
                                ard=True,
                                learn_scale=False,
                                Y=Y,
                                Poisson=Poisson)

    #latent distribution is itself a GP
    #     lat_dist = mgp.rdist.lat_GP(manif,
    #                                 m,
    #                                 n_samples,
    #                                 torch.Tensor(ts).to(device),
    #                                 Y=Y,
    #                                 initialization='fa')

    lat_dist = mgp.rdist.fast_EP_GP(manif,
                                    m,
                                    n_samples,
                                    torch.Tensor(ts).to(device),
                                    Y=Y,
                                    initialization='fa')

    ###construct prior
    lprior = mgp.lpriors.GP_full(dfit, m, n_samples, manif,
                                 torch.Tensor(ts).to(device))
    lprior = mgp.lpriors.Null(manif)

    # generate model
    if Poisson:
        likelihood = mgp.likelihoods.NegativeBinomial(n, Y=Y)
        #likelihood = mgp.likelihoods.Poisson(n)
    else:
        likelihood = mgp.likelihoods.Gaussian(n, Y=Y, d=dfit)
    z = manif.inducing_points(n, n_z)
    mod = mgp.models.SvgpLvm(n, m, n_samples, z, kernel, likelihood, lat_dist,
                             lprior).to(device)

    print(mod.lat_dist.name)
    ### test that training runs ###
    n_mc = 16

    def cb(mod, i, loss):
        if i % 5 == 0:
            print('')
        return

    mgp.optimisers.svgp.fit(data,
                            mod,
                            optimizer=optim.Adam,
                            n_mc=n_mc,
                            max_steps=5,
                            burnin=50,
                            lrate=10e-2,
                            print_every=5,
                            stop=cb,
                            analytic_kl=True)

    try:
        print('lat ell, scale:',
              mod.lat_dist.f.ell.detach().flatten(),
              mod.lat_dist.f.scale.detach().flatten())
        print('prior:', mod.lprior.ell.detach().flatten())
    except:
        print('lat ell, scale:',
              mod.lat_dist.ell.detach().flatten(),
              mod.lat_dist.scale.detach().mean(0).mean(-1))

    #print('kernel:', (mod.kernel.input_scale.detach())**(-1))
    mus = mod.lat_dist.prms[0].detach().cpu().numpy()
    print('mus:', np.std(mus, axis=(0, 1)))

    print(mus.shape)


#     plt.figure()
#     plt.hist(mus.flatten(), bins=30)
#     plt.savefig('figures/test_hist.png', bbox_inches='tight')
#     plt.close()

#     ex = 0
#     xs = xs[ex, ...] - np.mean(xs[ex, ...], axis=0, keepdims=True)
#     mus = mus[ex, ...] - np.mean(mus[ex, ...], axis=0, keepdims=True)

#     #align
#     if d == 1:
#         mus = mus[:, :1]

#     #fit xs = mus @ T --> T = (mus' * mus)^(-1) * mus' * xs
#     T = np.linalg.inv(mus.T @ mus) @ mus.T @ xs
#     mus = mus @ T  #predicted values

#     print(mus.shape)
#     plt.figure()
#     if d >= 2:
#         plt.plot(mus[:, 0], mus[:, 1], 'k-')
#         plt.plot(xs[:, 0], xs[:, 1], 'b-')
#     else:
#         plt.plot(ts[0, 0, :], mus[:, 0], 'k-')
#         plt.plot(ts[0, 0, :], xs[:, 0], 'b-')
#     plt.xlabel('lat dim 1')
#     plt.ylabel('lat dim 2')
#     plt.legend(['inferred', 'true'], frameon=False)
#     plt.savefig('figures/test_gp.png', bbox_inches='tight')
#     plt.close()

if __name__ == '__main__':
    test_fast_GP_lat_prior()
