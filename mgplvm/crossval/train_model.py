import mgplvm
from mgplvm import optimisers
import numpy as np
import pickle
import torch
from torch import optim


def training_params(**kwargs):

    params = {
        'max_steps': 1001,
        'burnin': 150,
        'callback': None,
        'optimizer': optim.Adam,
        'batch_size': None,
        'ts': None,
        'print_every': 50,
        'lrate': 5E-2,
        'batch_pool': None,
        'neuron_idxs': None,
        'mask_Ts': None,
        'n_mc': 32,
        'prior_m': None,
        'analytic_kl': False,
        'accumulate_gradient': True
    }

    for key, value in kwargs.items():
        if key in params.keys():
            params[key] = value
        else:
            print('adding', key)

    return params


def train_model(mod, data, params):

    dataloader = optimisers.data.BatchDataLoader(
        data, batch_size=params['batch_size'], batch_pool=params['batch_pool'])

    trained_mod = optimisers.svgp.fit(
        dataloader,
        mod,
        optimizer=params['optimizer'],
        max_steps=int(round(params['max_steps'])),
        burnin=params['burnin'],
        n_mc=params['n_mc'],
        lrate=params['lrate'],
        print_every=params['print_every'],
        stop=params['callback'],
        neuron_idxs=params['neuron_idxs'],
        mask_Ts=params['mask_Ts'],
        prior_m=params['prior_m'],
        analytic_kl=params['analytic_kl'],
        accumulate_gradient=params['accumulate_gradient'])

    return trained_mod
