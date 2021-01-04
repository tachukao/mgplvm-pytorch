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

#def initialize(Y,
#               device,
#               d,
#               n,
#               m,
#               n_z,
#               fit_manif=Torus,
#               GPparams=None,
#               Ntrain=None,
#               Tfix=slice(0),
#               sig0=1.5,
#               ell0=2):
#
#    if fit_manif == So3:
#        sig0 = 0.4  # sqrt diagonal covariance matrix
#    elif fit_manif == Torus:
#        sig0 = np.pi / 2
#    else:
#        sig0 = 1
#
#    if GPparams is None:
#
#        gammas = None
#        mu = None
#        ell = np.ones(n) * ell0
#        alpha = np.mean(np.std(Y, axis=1), axis=1)
#        sigma = np.mean(np.std(Y, axis=1), axis=1)  # initialize noise
#        z = None
#    else:
#        # get raw params since we don't have inverse map
#        mu = GPparams.manif.mu.data.cpu().numpy()
#
#        # get .prms since we initialize with inv_transform
#        gammas = GPparams.rdist.prms.data.cpu().numpy()
#        alpha, ell = [
#            prm.data.cpu().numpy()[Ntrain] for prm in GPparams.kernel.prms
#        ]
#        sigma, z = [prm.data.cpu()[Ntrain, :, :] for prm in GPparams.sgp.prms]
#        sigma = sigma[:, 0, 0]
#
#    # construct model
#    manif = fit_manif(m, d, mu=mu, Tinds=Tfix)
#    ref_dist = rdist.MVN(m, d, sigma=sig0, gammas=gammas, Tinds=Tfix)
#    kernel = kernels.QuadExp(n, manif.distance, alpha=alpha, ell=ell)
#    mod = models.Sgp(kernel, n, m, z=z, sigma=sigma).to(device)
#
#    return mod
#
#
#def recover_model(fname, device):
#    params = pickle.load(open(fname + '.pickled', 'rb'))
#    manifdict = {'Torus': Torus, 'Euclid': Euclid, 'So3': So3}
#    kerneldict = {'QuadExp': kernels.QuadExp}
#    rdistdict = {'MVN': mgplvm.rdist.MVN}
#    moddict = {'Sgp': models.Sgp}
#    manif = params['manif'].split('(')[0]
#    manif = 'So3' if manif == 'So' else manif
#    m, n, d, n_z = [params[key] for key in ['m', 'n', 'd', 'n_z']]
#    manif = manifdict[manif](m, d)
#    kernel = kerneldict[params['kernel']](n, manif.distance)
#    ref_dist = rdistdict[params['rdist']](m, d)
#    mod = moddict[params['model']](kernel, manif, n, m, n_z, kernel, ref_dist)
#    mod_params = torch.load(fname + '.torch')
#    mod.load_state_dict(mod_params)
#    mod.to(device)
#    return mod, params
#
#
#def train_cv(Y,
#             manifs,
#             n_z,
#             device,
#             callback=None,
#             max_steps=500,
#             n_b=128,
#             lrate=5e-2,
#             randN=True,
#             frac=2,
#             outname='test',
#             sig0=1.5,
#             ell0=2,
#             burnin='default'):
#    '''
#    given a dataset Y and a set of manifolds, fit each manifold to the data
#    manifs is a list of (manif, d)
#    frac is the inverse fraction of neurons used in the test set
#    '''
#    def trainfunc(Y, mod, burnin, trainGP=True, Tfix=slice(0), callback=None):
#        nbatch = 1
#        nbatch_max = 100
#        while nbatch < nbatch_max:
#            if nbatch > 1:
#                print('nbatch = ' + str(nbatch))
#            try:
#                return training.sgp(Y,
#                                    mod,
#                                    device,
#                                    trainGP=trainGP,
#                                    Tfix=Tfix,
#                                    max_steps=max_steps,
#                                    n_b=n_b,
#                                    callback=callback,
#                                    lrate=lrate,
#                                    burnin=burnin,
#                                    nbatch=nbatch)
#            except RuntimeError:
#                nbatch += 1
#
#        raise RuntimeError('maximum batch size exceeded')
#
#    try:
#        os.mkdir(outname)
#    except FileExistsError:
#        print(outname, 'already exists')
#
#    n, m = Y.shape[:2]
#    m1, n1 = int(m - m / frac), int(n - n / frac)  # 'test'
#
#    # random shuffle of timepoints
#    Tshuff = np.random.permutation(np.arange(m))
#    T1, T2 = Tshuff[:m1], Tshuff[m1:]
#
#    # random shuffle of neurons
#    Nshuff = np.random.permutation(np.arange(n)) if randN else np.arange(n)
#    N1, N2 = Nshuff[:n1], Nshuff[n1:]
#
#    Y1, Y2, Y3 = Y[:, T1], Y[N1, :], Y[N2, :]
#
#    params = {'Y': Y, 'N1': N1, 'N2': N2, 'T1': T1, 'T2': T2}
#    for i, (fit_manif, d) in enumerate(manifs):
#
#        print('\nfitting manifold', fit_manif(m, d).name)
#
#        if (burnin != 'default'):
#            burn = int(round(burnin / 3))
#        else:
#            burn = burnin
#
#        # fit all neurons half timepoints
#        mod1 = initialize(Y1,
#                          device,
#                          d,
#                          n,
#                          m1,
#                          n_z,
#                          fit_manif=fit_manif,
#                          sig0=sig0,
#                          ell0=ell0)
#        trainfunc(Y1, mod1, burn)
#        mod1.store_model(outname + '/' + str(i) + '_mod1', extra_params=params)
#
#        # fit all timepoints half neurons
#        mod2 = initialize(Y2,
#                          device,
#                          d,
#                          n1,
#                          m,
#                          n_z,
#                          fit_manif=fit_manif,
#                          GPparams=mod1,
#                          Ntrain=N1,
#                          Tfix=T1,
#                          sig0=sig0,
#                          ell0=ell0)
#        trainfunc(Y2, mod2, burn, trainGP=False, Tfix=T1, callback=callback)
#        mod2.store_model(outname + '/' + str(i) + '_mod2')
#        del mod2
#        torch.cuda.empty_cache()
#
#        # fit all timepoints half neurons reverse
#        mod3 = initialize(Y3,
#                          device,
#                          d,
#                          n1,
#                          m,
#                          n_z,
#                          fit_manif=fit_manif,
#                          GPparams=mod1,
#                          Ntrain=N2,
#                          Tfix=T1,
#                          sig0=sig0,
#                          ell0=ell0)
#        if frac == 2:
#            trainfunc(Y3,
#                      mod3,
#                      burn,
#                      trainGP=False,
#                      Tfix=T1,
#                      callback=callback)
#
#        mod3.store_model(outname + '/' + str(i) + '_mod3')
#
#        del mod1
#        del mod3
#
#        torch.cuda.empty_cache()
#
#    return params
#
#
#def gen_cvmodels(fbase, device, Type='MSE'):
#
#    mod1, params1 = recover_model(fbase + '1', device)
#    mod2, params2 = recover_model(fbase + '2', device)
#    mod3, params3 = recover_model(fbase + '3', device)
#
#    T1, T2 = [params1[key] for key in ['T1', 'T2']]
#
#    if Type == 'MSE':
#        mus2_T2 = mod2.manif.prms[T2, ...].detach()
#        mus3_T2 = mod3.manif.prms[T2, ...].detach()
#        for mod in [mod2, mod3]:
#            # change variational parameters mu, gamma to reference
#            mod.manif.mu = torch.nn.Parameter(mod1.manif.mu.detach())
#            mod.rdist.gamma = torch.nn.Parameter(mod1.rdist.gamma.detach())
#            mod.rdist.m, mod.manif.m, mod.m, mod.sgp.m = [
#                len(T1) for _ in range(4)
#            ]
#
#        return mod1, mod2, mod3, params1, mus2_T2, mus3_T2
#
#    else:
#        mus = [mod.manif.mu[T2, ...].detach() for mod in [mod3, mod2]]
#        gammas = [mod.rdist.gamma[T2, ...].detach() for mod in [mod3, mod2]]
#
#        # swap variational distributions
#        for mod, mu, gamma in zip([mod2, mod3], mus, gammas):
#            mod.manif.mu = torch.nn.Parameter(mu)
#            mod.rdist.gamma = torch.nn.Parameter(gamma)
#            mod.rdist.m, mod.manif.m, mod.m, mod.sgp.m = [
#                len(T2) for _ in range(4)
#            ]
#
#        return mod1, mod2, mod3, params1
#
#
#def calc_NLLs(fname, device=None, itermax='none', itermin=0, twoway=True):
#    iterdirs = np.sort(os.listdir(fname))
#    iterdirs = iterdirs if itermax == 'none' else iterdirs[:itermax]
#    iterdirs = iterdirs[itermin:]
#    niter = len(iterdirs)
#    device = mgplvm.utils.get_device() if device is None else device
#    print('\ncomputing cross-validated log likelihoods')
#
#    for (i_iter, iterdir) in enumerate(iterdirs):
#        manifs = np.sort(os.listdir(fname + "/" + iterdir))
#        nmanif = np.amax([int(f[0]) for f in manifs]) + 1
#
#        if i_iter == 0:
#            NLLs = np.zeros((niter, nmanif))
#            print(niter, 'iterations &', nmanif, 'manifolds')
#
#        for i_manif in range(nmanif):
#
#            mod1, mod2, mod3, params = gen_cvmodels(
#                fname + "/" + iterdir + '/' + str(i_manif) + '_mod',
#                device,
#                Type='LL')
#            Y, N1, N2, T2 = [params[key] for key in ['Y', 'N1', 'N2', 'T2']]
#            Y2, Y3 = Y[N1, :, :], Y[N2, :, :]
#            data2, data3 = [
#                torch.tensor(Ytest[:, T2, :], dtype=torch.get_default_dtype())
#                for Ytest in [Y2, Y3]
#            ]
#
#            # calc LL and MSE
#            LL2 = mod2.calc_LL(data2.to(device),
#                               128).data.cpu().numpy()  # trained on Y2
#            LL3 = mod3.calc_LL(data3.to(device),
#                               128).data.cpu().numpy()  # trained on Y3
#
#            if twoway:
#                NLL = -(LL2 + LL3) / 2
#            else:
#                NLL = -LL2
#
#            NLLs[i_iter, i_manif] = NLL
#            print(str(i_iter) + ':', mod1.manif.name, 'NLL=' + str(NLL))
#
#    return NLLs
#
#
#def calc_MSEs(fname,
#              device=None,
#              itermax='none',
#              iterp=100,
#              itermin=0,
#              twoway=True):
#
#    print('\ncomputing cross-validated mean squared errors')
#
#    iterdirs = np.sort(os.listdir(fname))
#    iterdirs = iterdirs if itermax == 'none' else iterdirs[:itermax]
#    iterdirs = iterdirs[itermin:]
#    niter = len(iterdirs)
#    device = mgplvm.utils.get_device() if device is None else device
#
#    for (i_iter, iterdir) in enumerate(iterdirs):
#        manifs = np.sort(os.listdir(fname + "/" + iterdir))
#        nmanif = np.amax([int(f[0]) for f in manifs]) + 1
#
#        if i_iter == 0:
#            MSEs = np.zeros((niter, nmanif))
#            print(niter, 'iterations &', nmanif, 'manifolds')
#
#        for i_manif in range(nmanif):
#
#            mod1, mod2, mod3, params, mus2_T2, mus3_T2 = gen_cvmodels(
#                fname + "/" + iterdir + '/' + str(i_manif) + '_mod',
#                device,
#                Type='MSE')
#
#            Y, T1, T2, N1, N2 = [
#                params[key] for key in ['Y', 'T1', 'T2', 'N1', 'N2']
#            ]
#            Y2, Y3 = Y[N1, :, :], Y[N2, :, :]
#
#            data2, data3 = [
#                torch.tensor(Ytrain[:, T1, :],
#                             dtype=torch.get_default_dtype()).to(device)
#                for Ytrain in [Y2, Y3]
#            ]
#
#            # trained on T1 (data), predict on T2 (manif.mu)
#            mus3_T2 = mus3_T2.to(device)
#            fmean2, _ = mod2.predict(data2, mus3_T2, niter=iterp)
#            fmean3, _ = mod3.predict(data3, mus2_T2, niter=iterp)
#            MSE2 = np.mean((fmean2.cpu().numpy() - Y2[:, T2, :])**2)
#            MSE3 = np.mean((fmean3.cpu().numpy() - Y3[:, T2, :])**2)
#
#            var2 = np.mean(np.var(Y2[:, T2, 0], axis=1))
#            var3 = np.mean(np.var(Y3[:, T2, 0], axis=1))
#
#            if twoway:
#                MSE = (MSE2 + MSE3) / 2
#            else:
#                MSE = MSE3
#
#            MSEs[i_iter, i_manif] = MSE
#            print(str(i_iter) + ':', mod1.manif.name, MSE, (var2 + var3) / 2)
#
#            for mod in [mod1, mod2, mod3]:
#                del mod
#            torch.cuda.empty_cache()
#
#    return MSEs
#