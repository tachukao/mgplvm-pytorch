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
        'n_mc': 32
    }

    for key, value in kwargs.items():
        params[key] = value

    return params


def train_model(mod, Y, device, params):

    trained_mod = optimisers.svgp.fit(Y,
                                      mod,
                                      device,
                                      optimizer=params['optimizer'],
                                      max_steps=int(round(
                                          params['max_steps'])),
                                      burnin=params['burnin'],
                                      n_mc=params['n_mc'],
                                      lrate=params['lrate'],
                                      print_every=params['print_every'],
                                      batch_size=params['batch_size'],
                                      stop=params['callback'],
                                      batch_pool=params['batch_pool'],
                                      neuron_idxs=params['neuron_idxs'],
                                      mask_Ts=params['mask_Ts']),

    return trained_mod