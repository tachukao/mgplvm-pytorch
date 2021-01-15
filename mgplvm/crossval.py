import os
import numpy as np
import mgplvm
import torch
from mgplvm import kernels, rdist, models, training
from mgplvm.manifolds import Torus, Euclid, So3
import matplotlib.pyplot as plt
import pickle
from scipy.stats import ttest_1samp
torch.set_default_dtype(torch.float64)


def not_in(arr, inds):
    mask = np.ones(arr.size, dtype=bool)
    mask[inds] = False
    return arr[mask]


def load_from(mod1, mod2, Ts, Ns):
    '''
    copy parameters from mod2 to mod1 for the condition indices given by Ts
    copy parameters for the neuron indices given by Ns
    '''
    ##### consider variational distribution #####

    ##### consider kernel #####

    ##### consider likelihood #####

    ##### consider prior #####

    return mod


def update_params(params, **kwargs):
    newps = copy.copy(params)
    for key, value in kwargs.items():
        newps[key] = value
    return newps


def train_cv(Y,
             device,
             model_ps,
             train_ps,
             outname,
             T1=None,
             N1=None,
             nt_train=None,
             nn_train=None,
             assess=True,
             save=True):
    '''
    given a dataset Y and a set of manifolds, fit each manifold to the data
    manifs is a list of (manif, d)
    frac is the inverse fraction of neurons used in the test set
    '''

    n, m = Y.shape[:2]
    nt_train = int(round(m / 2)) if nt_train is None else nt_train
    nn_train = int(round(n / 2)) if nn_train is None else nn_train

    if T1 is None:  # random shuffle of timepoints
        T1 = np.random.permutation(np.arange(m))[nt_train]
    if N1 is None:  # random shuffle of neurons
        N1 = np.random.permutation(np.arange(n))[nn_train]
    Y1, Y2 = Y[:, T1], Y[N1, :]
    split = {'Y': Y, 'N1': N1, 'T1': T1}

    # fit all neurons half timepoints
    model_ps1 = update_params(model_ps, Y=Y1, m=nt_train)
    mod1 = load_model(model_ps1)
    mod1 = train_model(mod1, Y, device, training_ps)
    if save: torch.save(mod1, outname + '_Ttrain.pt')

    # fit all timepoints half neurons
    ### need to fix all non-neuron parameters!!
    model_ps2 = update_params(model_ps, Y=Y2, n=nn_train)
    mod2 = load_model(model_ps2)

    mask = qq  #construct gradient mask

    load_from(mod2, mod1, T1, N1)
    mod2 = train_model(mod2, Y, device, training_ps)
    if save: torch.save(mod2, outname + '_Ntrain.pt')
    pickle.dump(split, open(outname + '_split.pickled', 'wb'))

    if assess:
        ##### assess the CV quality ####
        return

    else:
        return mod1, mod2


def gen_cvmodels(device, fbase=None, cv_data=None, Type='MSE'):

    if cv_data is None:
        mod1, mod2 = [
            torch.load(fbase + ext + 'train.pt') for ext in ['_T', '_N']
        ]
        split = pickle.load(open(fbase + '_split.pickled', 'rb'))
    else:
        mod1, mod2, split = cv_data

    n, m = split['Y'].shape[:2]
    T1, N1 = split['T1'], split['N1']
    T2, N2 = not_in(np.arange(m), T1), not_in(np.arange(n), N1)

    if Type == 'MSE':
        mus2_T2 = mod2.manif.prms[T2, ...].detach()
        mus3_T2 = mod3.manif.prms[T2, ...].detach()
        for mod in [mod2, mod3]:
            # change variational parameters mu, gamma to reference
            mod.manif.mu = torch.nn.Parameter(mod1.manif.mu.detach())
            mod.rdist.gamma = torch.nn.Parameter(mod1.rdist.gamma.detach())
            mod.rdist.m, mod.manif.m, mod.m, mod.sgp.m = [
                len(T1) for _ in range(4)
            ]

        return mod1, mod2, mod3, params1, mus2_T2, mus3_T2

    else:
        mus = [mod.manif.mu[T2, ...].detach() for mod in [mod3, mod2]]
        gammas = [mod.rdist.gamma[T2, ...].detach() for mod in [mod3, mod2]]

        # swap variational distributions
        for mod, mu, gamma in zip([mod2, mod3], mus, gammas):
            mod.manif.mu = torch.nn.Parameter(mu)
            mod.rdist.gamma = torch.nn.Parameter(gamma)
            mod.rdist.m, mod.manif.m, mod.m, mod.sgp.m = [
                len(T2) for _ in range(4)
            ]

        return mod1, mod2, mod3, params1


def calc_NLLs(fname, device=None, itermax='none', itermin=0, twoway=True):
    iterdirs = np.sort(os.listdir(fname))
    iterdirs = iterdirs if itermax == 'none' else iterdirs[:itermax]
    iterdirs = iterdirs[itermin:]
    niter = len(iterdirs)
    device = mgplvm.utils.get_device() if device is None else device
    print('\ncomputing cross-validated log likelihoods')

    for (i_iter, iterdir) in enumerate(iterdirs):
        manifs = np.sort(os.listdir(fname + "/" + iterdir))
        nmanif = np.amax([int(f[0]) for f in manifs]) + 1

        if i_iter == 0:
            NLLs = np.zeros((niter, nmanif))
            print(niter, 'iterations &', nmanif, 'manifolds')

        for i_manif in range(nmanif):

            mod1, mod2, mod3, params = gen_cvmodels(
                fname + "/" + iterdir + '/' + str(i_manif) + '_mod',
                device,
                Type='LL')
            Y, N1, N2, T2 = [params[key] for key in ['Y', 'N1', 'N2', 'T2']]
            Y2, Y3 = Y[N1, :, :], Y[N2, :, :]
            data2, data3 = [
                torch.tensor(Ytest[:, T2, :], dtype=torch.get_default_dtype())
                for Ytest in [Y2, Y3]
            ]

            # calc LL and MSE
            LL2 = mod2.calc_LL(data2.to(device),
                               128).data.cpu().numpy()  # trained on Y2
            LL3 = mod3.calc_LL(data3.to(device),
                               128).data.cpu().numpy()  # trained on Y3

            if twoway:
                NLL = -(LL2 + LL3) / 2
            else:
                NLL = -LL2

            NLLs[i_iter, i_manif] = NLL
            print(str(i_iter) + ':', mod1.manif.name, 'NLL=' + str(NLL))

    return NLLs


def calc_MSEs(fname,
              device=None,
              itermax='none',
              iterp=100,
              itermin=0,
              twoway=True):

    print('\ncomputing cross-validated mean squared errors')

    iterdirs = np.sort(os.listdir(fname))
    iterdirs = iterdirs if itermax == 'none' else iterdirs[:itermax]
    iterdirs = iterdirs[itermin:]
    niter = len(iterdirs)
    device = mgplvm.utils.get_device() if device is None else device

    for (i_iter, iterdir) in enumerate(iterdirs):
        manifs = np.sort(os.listdir(fname + "/" + iterdir))
        nmanif = np.amax([int(f[0]) for f in manifs]) + 1

        if i_iter == 0:
            MSEs = np.zeros((niter, nmanif))
            print(niter, 'iterations &', nmanif, 'manifolds')

        for i_manif in range(nmanif):

            mod1, mod2, mod3, params, mus2_T2, mus3_T2 = gen_cvmodels(
                fname + "/" + iterdir + '/' + str(i_manif) + '_mod',
                device,
                Type='MSE')

            Y, T1, T2, N1, N2 = [
                params[key] for key in ['Y', 'T1', 'T2', 'N1', 'N2']
            ]
            Y2, Y3 = Y[N1, :, :], Y[N2, :, :]

            data2, data3 = [
                torch.tensor(Ytrain[:, T1, :],
                             dtype=torch.get_default_dtype()).to(device)
                for Ytrain in [Y2, Y3]
            ]

            # trained on T1 (data), predict on T2 (manif.mu)
            mus3_T2 = mus3_T2.to(device)
            fmean2, _ = mod2.predict(data2, mus3_T2, niter=iterp)
            fmean3, _ = mod3.predict(data3, mus2_T2, niter=iterp)
            MSE2 = np.mean((fmean2.cpu().numpy() - Y2[:, T2, :])**2)
            MSE3 = np.mean((fmean3.cpu().numpy() - Y3[:, T2, :])**2)

            var2 = np.mean(np.var(Y2[:, T2, 0], axis=1))
            var3 = np.mean(np.var(Y3[:, T2, 0], axis=1))

            if twoway:
                MSE = (MSE2 + MSE3) / 2
            else:
                MSE = MSE3

            MSEs[i_iter, i_manif] = MSE
            print(str(i_iter) + ':', mod1.manif.name, MSE, (var2 + var3) / 2)

            for mod in [mod1, mod2, mod3]:
                del mod
            torch.cuda.empty_cache()

    return MSEs
