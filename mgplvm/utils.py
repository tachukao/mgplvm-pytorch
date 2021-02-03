import torch
from typing import Optional

default_jitter = 1E-8


def softplus(x):
    return torch.log(1 + torch.exp(x))


def inv_softplus(x):
    return torch.log(torch.exp(x) - 1)


def get_device(device: str = "cuda"):
    if torch.cuda.is_available() and device == "cuda":
        mydevice = torch.device(device)
    else:
        mydevice = torch.device('cpu')
        # need to allow multiple instances of openMP
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    return mydevice
