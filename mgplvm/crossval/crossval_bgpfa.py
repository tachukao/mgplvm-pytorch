import numpy as np
import copy
import torch
from .train_model import train_model
from .crossval import not_in, update_params
from ..manifolds import Euclid
from ..likelihoods import Gaussian, NegativeBinomial
from ..rdist import GP_circ, GP_diag
from ..lpriors import Null
from ..models import Lvgplvm

def train_cv_bgpfa(Y,
             device,
             train_ps,
             fit_ts,
            d_fit,
            ell,
             T1=None,
             N1=None,
             nt_train=None,
             nn_train=None,
             test=True,
            lat_scale = 1,
            rel_scale = 1):
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




    first construct one model then save parameters and store a new model copying over the generative params
    """
    
    print('training')
    
    _, n, m = Y.shape
    data = torch.tensor(Y, device=device, dtype=torch.get_default_dtype())
    nt_train = int(round(m / 2)) if nt_train is None else nt_train
    nn_train = int(round(n / 2)) if nn_train is None else nn_train

    if T1 is None:  # random shuffle of timepoints
        T1 = np.random.permutation(np.arange(m))[:nt_train]
    if N1 is None:  # random shuffle of neurons
        N1 = np.random.permutation(np.arange(n))[:nn_train]
    split = {'Y': Y, 'N1': N1, 'T1': T1}
    
    
    ##### fit the first model!!!! ####
    Y1 = Y[..., T1]
    n_samples, n, T = Y1.shape
    
    manif = Euclid(T, d_fit)
    lprior = Null(manif)
    lat_dist = GP_circ(manif, T, n_samples, fit_ts[..., T1], _scale=lat_scale, ell = ell) #initial ell ~200ms
    lik = Gaussian(n, Y=Y1, d = d_fit)
    
    mod = Lvgplvm(n, T, d_fit, n_samples, lat_dist, lprior, lik, ard = True, learn_scale = False, Y = Y1, rel_scale = rel_scale).to(device)
    
    train_model(mod, torch.tensor(Y1).to(device), train_ps) ###initial training####
    
    ### fit second model ###
    #### parameters to initialize:
    Y2 = Y
    n_samples, n, T = Y2.shape
    
    ###rdist: ell
    ell0 = mod.lat_dist.ell.detach().cpu()
    lat_dist = GP_circ(manif, T, n_samples, fit_ts, _scale=lat_scale, ell = ell0) #initial ell ~200ms
    
    ###lik: sigma
    sigma = mod.obs.likelihood.sigma.detach().cpu()
    lik = Gaussian(n, sigma = sigma)

    ###obs: q_mu, q_sqrt, _scale, _dim_scale, _neuron_scale
    q_mu, q_sqrt = mod.obs.q_mu.detach().cpu(), mod.obs.q_sqrt.detach().cpu()
    scale, dim_scale, neuron_scale = mod.obs.scale.detach().cpu(), mod.obs.dim_scale.detach().cpu().flatten(), mod.obs.neuron_scale.detach().cpu().flatten()
    mod = Lvgplvm(n, T, d_fit, n_samples, lat_dist, lprior, lik, ard = True, learn_scale = False,
                q_mu = q_mu, q_sqrt = q_sqrt, scale = scale, dim_scale = dim_scale, neuron_scale = neuron_scale).to(device)
    torch.cuda.empty_cache
    
    for p in mod.parameters():  #no gradients for the remaining parameters
        p.requires_grad = False

    mod.lat_dist.nu.requires_grad = True #latent variational mean
    mod.lat_dist._scale.requires_grad = True #latent variational covariance
    if 'circ' in mod.lat_dist.name:
        mod.lat_dist._c.requires_grad = True #latent variational covariance

    train_ps2 = update_params(train_ps, neuron_idxs=N1, max_steps = int(round(train_ps['max_steps'])))
    train_model(mod, torch.tensor(Y2).to(device), train_ps2)

    if test:
        test_cv(mod, split, device, n_mc=train_ps['n_mc'], Print=True)

    return mod, split
