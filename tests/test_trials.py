import numpy as np
import torch
from torch import optim
import mgplvm
from mgplvm import kernels, rdist, models, optimisers, syndata, likelihoods
from mgplvm.manifolds import Torus, Euclid, So3
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def test_trial_structure():
    """
    test that svgp runs without explicit check for correctness
    also test that burda log likelihood runs and is smaller than elbo
    """
    d = 1  # dims of latent space
    n = 8  # number of neurons
    m = 10  # number of conditions / time points
    n_z = 5  # number of inducing points
    n_samples = 2  # number of samples
    Y = np.random.normal(0, 1, (n_samples, n, m))
    print(Y.shape)

    zs = torch.randn((n, d, n_z))
    sig0 = 0.1

    # specify manifold, kernel and rdist
    manif1 = Euclid(m, d)
    lat_dist1 = mgplvm.rdist.ReLie(manif1,
                                   m,
                                   n_samples,
                                   diagonal=False,
                                   initialization='pca',
                                   Y=Y,
                                   sigma=sig0)
    kernel1 = kernels.QuadExp(n, manif1.distance, Y=Y)
    lik1 = likelihoods.Gaussian(n)
    lprior1 = mgplvm.lpriors.Uniform(manif1)
    z1 = manif1.inducing_points(n, n_z, z=zs)
    mod1 = models.SvgpLvm(n,
                          m,
                          n_samples,
                          z1,
                          kernel1,
                          lik1,
                          lat_dist1,
                          lprior1,
                          whiten=True).to(device)

    Y2 = Y.transpose(1, 0, 2).reshape(n, -1)[None, ...]
    n_samples2, n2, m2 = Y2.shape
    print(Y2.shape)
    manif2 = Euclid(m2, d)
    lat_dist2 = mgplvm.rdist.ReLie(manif2,
                                   m2,
                                   n_samples2,
                                   diagonal=False,
                                   initialization='pca',
                                   Y=Y2,
                                   sigma=sig0)
    kernel2 = kernels.QuadExp(n2, manif2.distance, Y=Y2)
    lik2 = likelihoods.Gaussian(n2)
    lprior2 = mgplvm.lpriors.Uniform(manif2)
    z2 = manif2.inducing_points(n2, n_z, z=zs)
    mod2 = models.SvgpLvm(n2,
                          m2,
                          1,
                          z2,
                          kernel2,
                          lik2,
                          lat_dist2,
                          lprior2,
                          whiten=True).to(device)

    assert torch.allclose(mod1.svgp.kernel.prms[0], mod2.svgp.kernel.prms[0])
    assert torch.allclose(mod1.svgp.kernel.prms[1], mod2.svgp.kernel.prms[1])
    assert torch.allclose(mod1.svgp.prms[0], mod2.svgp.prms[0])
    assert torch.allclose(mod1.svgp.prms[1], mod2.svgp.prms[1])
    assert torch.allclose(mod1.svgp.prms[2], mod2.svgp.prms[2])
    assert torch.allclose(mod1.lat_dist.prms[0].reshape(-1, d),
                          mod2.lat_dist.prms[0].reshape(-1, d))
    assert torch.allclose(mod1.lat_dist.prms[1].reshape(-1, d, d),
                          mod2.lat_dist.prms[1].reshape(-1, d, d))
    assert torch.allclose(mod1.svgp.likelihood.prms, mod2.svgp.likelihood.prms)

    n_mc = 9
    print(mod1.forward(torch.tensor(Y).to(device), n_mc))
    print(mod2.forward(torch.tensor(Y2).to(device), n_mc))

    nrep = 20
    mod1s, mod2s = [np.zeros(nrep) for i in range(2)]
    for i in range(nrep):  #compute the LLs, should be similar
        mod1s[i] = mod1.forward(torch.tensor(Y).to(device),
                                n_mc)[0].detach().cpu().numpy()
        mod2s[i] = mod2.forward(torch.tensor(Y2).to(device),
                                n_mc)[0].detach().cpu().numpy()
    comp = min(np.sum(mod1s > mod2s), np.sum(mod2s > mod1s))
    print(comp)
    assert comp > 0  #basically check that one version is not consistently lower/higher than the other


if __name__ == '__main__':
    test_trial_structure()
