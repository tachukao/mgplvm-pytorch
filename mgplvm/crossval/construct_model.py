import mgplvm
from mgplvm import lpriors, kernels, models
from mgplvm.manifolds import Euclid, Torus, So3
import torch
import pickle
import numpy as np


def model_params(n, m, d, n_z, n_samples, **kwargs):

    params = {
        'n': n,
        'm': m,
        'd': d,
        'n_z': n_z,
        'n_samples': n_samples,
        'manifold': 'euclid',
        'kernel': 'RBF',
        'prior': 'Uniform',
        'likelihood': 'Gaussian',
        'initialization': 'pca',
        'Y': None,
        'latent_sigma': 1,
        'latent_mu': None,
        'diagonal': True,
        'learn_linear_weights': False,
        'learn_linear_alpha': True,
        'linear_alpha': None,
        'RBF_alpha': None,
        'RBF_ell': None,
        'arp_p': 1,
        'arp_eta': np.ones(d) * 0.3,
        'arp_learn_eta': True,
        'arp_learn_c': False,
        'arp_learn_phi': True,
        'lik_gauss_std': None,
        'ts': torch.arange(m)[None, None, ...].repeat(n_samples, 1, 1),
        'device': None
    }

    for key, value in kwargs.items():
        params[key] = value

    return params


def load_model(params):

    n, m, d, n_z, n_samples = params['n'], params['m'], params['d'], params[
        'n_z'], params['n_samples']

    #### specify manifold ####
    if params['manifold'] == 'euclid':
        manif = Euclid(m, d)
    elif params['manifold'] == 'torus':
        manif = Torus(m, d)
    elif params['manifold'] in ['SO3', 'So3', 'so3', 'SO(3)']:
        manif = So3(m, 3)
        params['diagonal'] = False

    #### specify latent distribution ####
    lat_dist = mgplvm.rdist.ReLie(manif,
                                  m,
                                  n_samples,
                                  sigma=params['latent_sigma'],
                                  diagonal=params['diagonal'],
                                  initialization=params['initialization'],
                                  Y=params['Y'],
                                  mu=params['latent_mu'])

    #### specify kernel ####
    if params['kernel'] == 'linear':
        kernel = kernels.Linear(n,
                                manif.linear_distance,
                                d,
                                learn_weights=params['learn_linear_weights'],
                                learn_alpha=params['learn_linear_alpha'],
                                Y=params['Y'],
                                alpha=params['linear_alpha'])
    elif params['kernel'] == 'RBF':
        ell = None if params['RBF_ell'] is None else np.ones(
            n) * params['RBF_ell']
        kernel = kernels.QuadExp(n,
                                 manif.distance,
                                 Y=params['Y'],
                                 alpha=params['RBF_alpha'],
                                 ell=ell)

    #### speciy prior ####
    if params['prior'] == 'GP':
        lprior_kernel = kernels.QuadExp(d,
                                        manif.distance,
                                        learn_alpha=False,
                                        ell=np.ones(d) * m / 10)
        lprior = lpriors.GP(d,
                            n_samples,
                            manif,
                            lprior_kernel,
                            n_z=n_z,
                            ts=params['ts'])
    elif params['prior'] == 'ARP':
        lprior = lpriors.ARP(params['arp_p'],
                             manif,
                             ar_eta=torch.tensor(params['arp_eta']),
                             learn_eta=params['arp_learn_eta'],
                             learn_c=params['arp_learn_c'],
                             diagonal=params['diagonal'])
    else:
        lprior = lpriors.Uniform(manif)

    #### specify likelihood ####
    if params['likelihood'] == 'Gaussian':
        var = None if params['lik_gauss_std'] is None else np.square(
            params['lik_gauss_std'])
        likelihood = mgplvm.likelihoods.Gaussian(n, variance=var)
    elif params['likelihood'] == 'Poisson':
        likelihood = mgplvm.likelihoods.Poisson(n)
    elif params['likelihood'] == 'NegBinom':
        likelihood = mgplvm.likelihoods.NegativeBinomial(n)

    #### specify inducing points ####
    z = manif.inducing_points(n, n_z)

    #### construct model ####
    device = (mgplvm.utils.get_device()
              if params['device'] is None else params['device'])
    mod = models.SvgpLvm(n, z, kernel, likelihood, lat_dist, lprior).to(device)

    return mod
