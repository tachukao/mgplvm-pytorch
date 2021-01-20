import torch


def conj(x):
    a = torch.tensor([1, -1, -1, -1]).to(x.device)
    return a * x


def product(x, y):
    x0 = x[..., 0]
    x1 = x[..., 1]
    x2 = x[..., 2]
    x3 = x[..., 3]
    y0 = y[..., 0]
    y1 = y[..., 1]
    y2 = y[..., 2]
    y3 = y[..., 3]
    z = ((x0 * y0) - (x1 * y1) - (x2 * y2) - (x3 * y3),
         (x0 * y1) + (x1 * y0) - (x2 * y3) + (x3 * y2),
         (x0 * y2) + (x1 * y3) + (x2 * y0) - (x3 * y1),
         (x0 * y3) - (x1 * y2) + (x2 * y1) + (x3 * y0))
    return torch.stack(z, dim=-1)
