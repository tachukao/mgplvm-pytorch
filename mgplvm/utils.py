import torch

default_jitter = 1E-8


def softplus(x):
    return torch.log(1 + torch.exp(x))


def inv_softplus(x):
    return torch.log(torch.exp(x) - 1)


def get_device(device="cuda"):
    if torch.cuda.is_available() and device == "cuda":
        device = torch.device(device)
    else:
        device = torch.device('cpu')
        # need to allow multiple instances of openMP
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    return device