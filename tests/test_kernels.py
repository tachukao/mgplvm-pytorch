import torch
from mgplvm.kernels import QuadExp, QuadExpARD
from mgplvm.manifolds import Euclid


def test_quad_exp_hyp_prms_dims():
    n = 10
    kernel = QuadExp(n, Euclid.distance)
    alpha, ell = kernel.prms
    assert (alpha.shape == (n,))
    assert (ell.shape == (n,))


def test_quad_expard_hyp_prms_dims():
    n = 10
    d = 3
    kernel = QuadExpARD(n, d, Euclid.distance_ard)
    alpha, ell = kernel.prms
    assert (alpha.shape == (n,))
    assert (ell.shape == (n, d))


def test_quad_expard_trK():
    n_b = 2
    n = 10
    m = 20
    d = 3
    kernel = QuadExp(n, Euclid.distance)
    x = torch.randn(n_b, n, d, m)
    trK1 = kernel.trK(x)
    trK2 = torch.diagonal(kernel(x, x), dim1=2, dim2=3).sum(-1)
    assert torch.allclose(trK1, trK2)


def test_quad_expard_diagK():
    n_b = 2
    n = 10
    m = 20
    d = 3
    kernel = QuadExp(n, Euclid.distance)
    x = torch.randn(n_b, n, d, m)
    diagK1 = kernel.diagK(x)
    diagK2 = torch.diagonal(kernel(x, x), dim1=2, dim2=3)
    assert torch.allclose(diagK1, diagK2)
