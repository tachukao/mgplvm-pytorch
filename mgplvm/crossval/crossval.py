import os
import numpy as np
import copy
import mgplvm
import torch
from mgplvm import kernels, rdist, models 
from mgplvm.manifolds import Torus, Euclid, So3
import matplotlib.pyplot as plt
import pickle
from scipy.stats import ttest_1samp
from . import train_model, load_model
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
            T1 = None,
            N1 = None,
            nt_train = None,
            nn_train = None,
            test = True):
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

    n_samples, n, m = Y.shape
    nt_train = int(round(m/2)) if nt_train is None else nt_train
    nn_train = int(round(n/2)) if nn_train is None else nn_train
    
    if T1 is None: # random shuffle of timepoints
        T1 = np.random.permutation(np.arange(m))[:nt_train]
    if N1 is None: # random shuffle of neurons
        N1 = np.random.permutation(np.arange(n))[:nn_train]
    Y1, Y2 = Y[:, :, T1], Y[:, N1, :]
    split = {'Y': Y, 'N1': N1, 'T1': T1}
    
    train_ps1 = update_params(train_ps, batch_pool = T1)
    _ = train_model(mod, Y, device, train_ps1)
    
    ### construct a mask for some of the time points ####
    def mask_Ts(grad):
        ''' used to 'mask' some gradients for cv'''
        grad[:, T1, ...] *= 0
        return grad
    train_ps2 = update_params(train_ps, neuron_idxs = N1, mask_Ts = mask_Ts)
    
    
    for p in mod.parameters(): #no gradients for the remaining parameters
        p.requires_grad = False
    for p in mod.lat_dist.parameters(): #only gradients for the latent distribution
        p.requires_grad = True
    
    _ = train_model(mod, Y, device, train_ps2)
    
    if test:
        _ = test_cv(mod, split, device, n_mc = train_ps['n_mc'], Print = True)
        
    return mod, split
        

def test_cv(mod, split, device, n_mc = 32, Print = False):
    Y, T1, N1 = split['Y'], split['T1'], split['N1']
    n_samples, n, m = Y.shape
    
    ##### assess the CV quality ####
    T2, N2 = not_in(np.arange(m), T1), not_in(np.arange(n), N1)

    #generate prediction for held out data#
    
    Ytest = Y[:, N2, :][..., T2] #(ntrial x N2 x T2)
    latents = mod.lat_dist.prms[0].detach()[:, T2, ...] #latent means (ntrial, T2, d)
    query = latents.transpose(-1,-2) #(ntrial, d, m)
    Ypred, var = mod.svgp.predict(query[None, ...], False)
    Ypred = Ypred.detach().cpu().numpy()[0, :, N2, :] #(ntrial, N2, T2)
    MSE = np.mean((Ypred - Ytest)**2)
    
    var_cap = 1-np.var(Ytest - Ypred)/np.var(Ytest)


    ### compute crossvalidated log likelihood ###
    #(n_mc, n_samples, n), (n_mc, n_samples)
    svgp_elbo, kl = mod.elbo(torch.tensor(Y).to(device), n_mc, batch_idxs=T2, neuron_idxs = N2)
    
    svgp_elbo = svgp_elbo.sum(-1).sum(-1) #(n_mc)
    LLs = svgp_elbo - kl.sum(-1)  # LL for each batch (n_mc, )
    LL = (torch.logsumexp(LLs, 0) - np.log(n_mc)).detach().cpu().numpy()
    LL = LL/(len(T2)*len(N2)*n_samples)
    
    if Print:
        print('LL', LL)
        print('var_cap', var_cap)
        print('MSE', MSE, np.sqrt(np.mean(np.var(Ytest, axis = -1))))
    
    return MSE, LL, var_cap
    
    