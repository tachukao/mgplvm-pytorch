from .. import lpriors, kernels, models, rdist, likelihoods, utils
from ..manifolds import Euclid, Torus, So3
from ..manifolds.base import Manifold
from ..likelihoods import Likelihood
from ..lpriors.common import Lprior
from ..kernels import Kernel
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
        'initialization': 'fa',
        'Y': None,
        'latent_sigma': 1,
        'latent_mu': None,
        'diagonal': True,
        'learn_linear_scale': True,
        'linear_scale': None,
        'RBF_scale': None,
        'RBF_ell': None,
        'arp_p': 1,
        'arp_eta': np.ones(d) * 1,
        'arp_learn_eta': True,
        'arp_learn_c': False,
        'arp_learn_phi': True,
        'prior_ell': None,
        'lik_gauss_std': None,
        'ts': torch.arange(m)[None, None, ...].repeat(n_samples, 1, 1),
        'device': None
    }

    for key, value in kwargs.items():
        if key in params.keys():
            params[key] = value
        else:
            print('key not found; adding', key)
            params[key] = value

    return params


def load_model(params):

    n, m, d, n_z, n_samples = params['n'], params['m'], params['d'], params[
        'n_z'], params['n_samples']
    device = (utils.get_device()
              if params['device'] is None else params['device'])

    #### specify manifold ####
    if params['manifold'] == 'euclid':
        manif: Manifold = Euclid(m, d)
    elif params['manifold'] == 'torus':
        manif = Torus(m, d)
    elif params['manifold'] in ['SO3', 'So3', 'so3', 'SO(3)']:
        manif = So3(m, 3)
        params['diagonal'] = False

    #### specify latent distribution ####
    lat_dist: rdist.Rdist = rdist.ReLie(manif,
                                        m,
                                        n_samples,
                                        sigma=params['latent_sigma'],
                                        diagonal=params['diagonal'],
                                        initialization=params['initialization'],
                                        Y=params['Y'],
                                        mu=params['latent_mu'])

    #### specify kernel ####
    if params['kernel'] == 'linear':
        kernel: Kernel = kernels.Linear(
            n,
            d,
            learn_scale=params['learn_linear_scale'],
            Y=params['Y'],
            scale=params['linear_scale'])
    elif params['kernel'] == 'RBF':
        ell = None if params['RBF_ell'] is None else np.ones(
            n) * params['RBF_ell']
        kernel = kernels.QuadExp(n,
                                 manif.distance,
                                 Y=params['Y'],
                                 scale=params['RBF_scale'],
                                 ell=ell)

    #### speciy prior ####
    if params['prior'] == 'GP':
        """
        lprior_kernel = kernels.QuadExp(d,
                                        manif.distance,
                                        learn_scale=False,
                                        ell=np.ones(d) * m / 10)
        lprior: Lprior = lpriors.GP(d,
                                    m,
                                    n_samples,
                                    manif,
                                    lprior_kernel,
                                    n_z=n_z,
                                    ts=params['ts'])
        """
        lat_dist = rdist.GP_diag(manif,
                                 m,
                                 n_samples,
                                 params['ts'].to(device),
                                 ell=params['prior_ell'])
        lprior: Lprior = lpriors.Null(manif)

    elif params['prior'] == 'ARP':
        lprior = lpriors.ARP(params['arp_p'],
                             manif,
                             ar_eta=torch.tensor(params['arp_eta']),
                             learn_eta=params['arp_learn_eta'],
                             learn_c=params['arp_learn_c'],
                             diagonal=params['diagonal'])
    elif params['prior'] == 'LDS':
        lprior = lpriors.DS(manif)
    else:
        lprior = lpriors.Uniform(manif)

    #### specify likelihood ####
    if params['likelihood'] == 'Gaussian':
        likelihood: Likelihood = likelihoods.Gaussian(
            n, sigma=params['lik_gauss_std'], Y=params['Y'], d=d)
    elif params['likelihood'] == 'Poisson':
        likelihood = likelihoods.Poisson(n)
    elif params['likelihood'] == 'NegBinom':
        likelihood = likelihoods.NegativeBinomial(n, Y=params['Y'])

    #### specify inducing points ####
    z = manif.inducing_points(n, n_z)

    #### construct model ####
    mod = models.SvgpLvm(n, m, n_samples, z, kernel, likelihood, lat_dist,
                         lprior).to(device)

    return mod
