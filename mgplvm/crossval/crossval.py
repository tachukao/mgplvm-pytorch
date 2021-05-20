import numpy as np
import copy
import torch
from .train_model import train_model

torch.set_default_dtype(torch.float64)


def not_in(arr, inds):
    mask = np.ones(arr.size, dtype=bool)
    mask[inds] = False
    return arr[mask]


def update_params(params, **kwargs):
    newps = copy.copy(params)
    for key, value in kwargs.items():
        newps[key] = value
    return newps


def train_cv(mod,
             Y,
             device,
             train_ps,
             T1=None,
             N1=None,
             nt_train=None,
             nn_train=None,
             test=True):
    """
    Parameters
    ----------
    mod : mgplvm.models.svgplvm
        instance of svgplvm model to perform crossvalidation on.
    Y : array
        data with dimensionality (n x m x n_samples)
    device : torch.device
        GPU/CPU device on which to run the calculations
    train_ps : dict
        dictionary of training parameters. Constructed by crossval.training_params()
    T1 : Optional[int list]
        indices of the conditions to use for training
    N1 : Optional[int list]
        indices of the neurons to use for training
    nt_train : Optional[int]
        number of randomly selected conditions to use for training
    nn_train : Optional[int]
        number of randomly selected neurons to use for training

    Returns
    -------
    mod : mgplvm.svgplvm
        model trained via crossvalidation

    """

    _, n, m = Y.shape
    data = torch.tensor(Y, device=device, dtype=torch.get_default_dtype())
    nt_train = int(round(m / 2)) if nt_train is None else nt_train
    nn_train = int(round(n / 2)) if nn_train is None else nn_train

    if T1 is None:  # random shuffle of timepoints
        T1 = np.random.permutation(np.arange(m))[:nt_train]
    if N1 is None:  # random shuffle of neurons
        N1 = np.random.permutation(np.arange(n))[:nn_train]
    split = {'Y': Y, 'N1': N1, 'T1': T1}

    train_ps1 = update_params(train_ps, batch_pool=T1, prior_m=len(T1))
    train_model(mod, data, train_ps1)

    for p in mod.parameters():  #no gradients for the remaining parameters
        p.requires_grad = False

    if 'GP' in mod.lat_dist.name:
        mod.lat_dist.nu.requires_grad = True
        mod.lat_dist._scale.requires_grad = True
        mask_Ts = None
    else:

        def mask_Ts(grad):
            ''' used to 'mask' some gradients for cv'''
            grad[:, T1, ...] *= 0
            return grad

        for p in mod.lat_dist.parameters(
        ):  #only gradients for the latent distribution
            p.requires_grad = True

    train_ps2 = update_params(train_ps,
                              neuron_idxs=N1,
                              mask_Ts=mask_Ts,
                              prior_m=None)

    train_model(mod, data, train_ps2)

    if test:
        test_cv(mod, split, device, n_mc=train_ps['n_mc'], Print=True)

    return mod, split


def test_cv(mod, split, device, n_mc=32, Print=False, sample_mean=False, sample_X = False):
    Y, T1, N1 = split['Y'], split['T1'], split['N1']
    n_samples, n, m = Y.shape

    ##### assess the CV quality ####
    T2, N2 = not_in(np.arange(m), T1), not_in(np.arange(n), N1)

    #generate prediction for held out data#

    Ytest = Y[:, N2, :][..., T2]  #(ntrial x N2 x T2)
    
    #latent means (ntrial, T2, d)
    if 'GP' in mod.lat_dist.name:
        latents = mod.lat_dist.lat_mu.detach()[:, T2, ...] 
    else:
        latents = mod.lat_dist.prms[0].detach()[:, T2, ...]  
        
    query = latents.transpose(-1, -2)  #(ntrial, d, m)

    if sample_X: #note this only works when the data is structured as a single trial!
        n_mc = round(np.sqrt(n_mc))
        # g is shape (n_samples, n_mc, m, d)
        g, lq = mod.lat_dist.sample(torch.Size([n_mc]),
                             torch.tensor(Y).to(device),
                             batch_idxs=None,
                             sample_idxs=None)
        print(g.shape)
        assert g.shape[1] == 1 #assume there is only a single 'trial'

        query = g[:, 0, ...].transpose(-1, -2) #now each sample is a 'trial'
        Ypred = mod.svgp.sample(query, n_mc=n_mc, noise=False)
        print(Ypred.shape)
        Ypred = Ypred.mean(0).mean(0) #average over both sets of MC samples
        Ypred = Ypred.detach().cpu().numpy()[N2, :][:, T2][None, ...]  #(1 x N2 x T2)
            
    elif sample_mean:  #we don't have a closed form mean prediction so sample from (mu|GP) and average instead
        #n_mc x n_samples x N x d
        Ypred = mod.svgp.sample(query, n_mc=n_mc, noise=False)
        Ypred = Ypred.mean(0).cpu().numpy()[:, N2, :]  #(ntrial x N2 x T2)
    else:
        Ypred, var = mod.svgp.predict(query[None, ...], False)
        Ypred = Ypred.detach().cpu().numpy()[0][:, N2, :]  #(ntrial, N2, T2)
    MSE_vals = np.mean((Ypred - Ytest)**2, axis=(0, -1))
    MSE = np.mean(MSE_vals)  #standard MSE
    norm_MSE = MSE_vals / np.var(Ytest,
                                 axis=(0, -1))  #normalize by neuron variance
    norm_MSE = np.mean(norm_MSE)

    #print('means:', np.mean(Ytest), np.mean(Ypred))
    var_cap = 1 - np.var(Ytest - Ypred) / np.var(Ytest)

    ### compute crossvalidated log likelihood ###
    #mold = mod.m
    #mod.m = len(T2) #use correct scaling factor for the test data
    #mod.svgp.m = len(T2)

    data = torch.tensor(Y, device=device)
    #(n_mc, n_samples, n), (n_mc, n_samples)
    svgp_elbo, kl = mod.elbo(data[:, :, T2],
                             n_mc,
                             batch_idxs=T2,
                             neuron_idxs=N2,
                             m=len(T2))

    #mod.m = mold #restore original scaling factor
    #mod.svgp.m = mold

    svgp_elbo = svgp_elbo.sum(-1)  #(n_mc)
    LLs = svgp_elbo - kl  # LL for each batch (n_mc, )
    LL = (torch.logsumexp(LLs, 0) - np.log(n_mc)).detach().cpu().numpy()
    LL = LL / (len(T2) * len(N2) * n_samples)

    if Print:
        print('LL', LL)
        print('var_cap', var_cap)
        print('MSE', MSE, np.sqrt(np.mean(np.var(Ytest, axis=-1))))

    return MSE, LL, var_cap, norm_MSE
