import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from torch import optim

import mgplvm
from mgplvm import kernels, rdist
from mgplvm.manifolds import Torus, Euclid, So3
from mgplvm.models import Core
from mgplvm.training import train
from mgplvm.utils import get_device

torch.set_default_dtype(torch.float64)
np.random.seed(9310207)
torch.manual_seed(9310207)


def fit_th1(nbatch=2, epoch='wake', manif=Torus, d=1, n_z=10, ell0=1.5, sig0=2):
    '''epoch is wake or sleep
    manif is the latent topology, d is the dimensionality.
    nbatch can be increased in case in case of insufficien RAM.
    n_z is number of inducing points'''

    device = get_device()
    print(device)

    data = pickle.load(open('./data/mouse/binned_data.pickled', 'rb'))
    if epoch == 'wake':
        Y, zs = data['Y_wake'], data['hd_wake']
    else:
        Y, zs = data['Y_sleep'], data['hd_sleep']

    n, m = Y.shape

    # sqrt normalize to stabilize variance
    Y = np.sqrt(Y)
    Y = Y - np.mean(Y, axis=1, keepdims=True)

    n_z = 10  # number of inducing points
    alpha = np.std(Y, axis=1)
    sigma = np.std(Y, axis=1)  # initialize noise

    manif = manif(m, d, mu=None)  # initialize mean at identity
    ref_dist = mgplvm.rdist.MVN(m, d, sigma=sig0)  # base distribution
    kernel = mgplvm.kernels.QuadExp(n,
                                    manif.distance,
                                    alpha=alpha,
                                    ell=np.ones(n) * ell0)
    mod = Core(manif, n, m, n_z, kernel, ref_dist, sigma=sigma).to(device)

    Y = Y.reshape(n, m, 1).astype(np.float64)  # only one sample

    print('fitting', manif.name, 'to', epoch)
    print('neurons:', n, '  timepoints: ', m)

    def callback(model, i):
        ''' plot progress during optimization '''
        if i % 10 == 0:
            manifs = model.manif if type(model.manif) == list else [model.manif]
            plt.figure()
            msize = 3
            g_mus = model.manif.prms.data.cpu().numpy()[:, 0]
            if epoch == 'wake':
                plt.plot(zs[:], g_mus, "ko", markersize=msize)
            else:
                plt.plot(np.arange(len(g_mus)),
                         np.sort(g_mus),
                         "ko",
                         markersize=msize)
            plt.xlabel('head direction')
            plt.ylabel('inferred latent')
            plt.title('i = ' + str(i))
            plt.savefig("yo")
            plt.close()

    try:
        # train model
        trained_mod = train(Y,
                            mod,
                            device,
                            optimizer=optim.Adam,
                            outdir='none',
                            max_steps=1000,
                            burnin=200,
                            n_b=128,
                            lrate=5E-2,
                            callback=callback,
                            nbatch=nbatch)
    except RuntimeError:
        print('out of memory, consider increasing nbatch from', nbatch)
        return

    # save model
    trained_mod.store_model('trained_models/mouse_' + epoch,
                            extra_params={
                                'Y': Y,
                                'zs': zs
                            })


if __name__ == '__main__':
    print('running')
    fit_th1(manif=Torus, epoch='wake', d=1)
