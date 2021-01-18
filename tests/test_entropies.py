import mgplvm
from mgplvm import rdist
from mgplvm.utils import get_device
from mgplvm.manifolds import So3, Torus, Euclid, S3
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
device = mgplvm.utils.get_device()


def test_euclid(kmax=5, savefig=False):
    m = 100
    for d in [1, 2, 3]:

        manif = Euclid(m, d)
        sigmas = 10**np.linspace(-2, 1, num=m).reshape(m, 1)
        q = mgplvm.rdist.ReLie(manif, m, sigma=torch.tensor(sigmas)).mvn()
        x = q.rsample(torch.Size([200]))
        lq = manif.log_q(q.log_prob, x, manif.d, kmax=kmax)
        H = -lq.mean(dim=0).detach().numpy()
        std = lq.std(dim=0).detach().numpy()

        Hgauss = d / 2 + d / 2 * np.log(2 * np.pi) + d * np.log(sigmas[:, 0])

        if savefig:
            plt.figure()
            plt.plot(sigmas[:, 0], Hgauss, 'g-')
            plt.plot(sigmas[:, 0], H, 'b--')
            plt.fill_between(sigmas[:, 0],
                             H - std,
                             H + std,
                             color='b',
                             alpha=0.2)
            plt.xscale('log')
            plt.savefig('test_euclid' + str(d) + '.png', dpi=120)
            plt.close()
        assert all(np.abs(H - Hgauss) < std)  # adhere to bound


def test_torus(kmax=5, savefig=False):

    m = 100

    for d in [1, 2, 3]:

        manif = Torus(m, d)
        sigmas = 10**np.linspace(-2, 1, num=m).reshape(m, 1)
        q = mgplvm.rdist.ReLie(manif, m, sigma=torch.tensor(sigmas)).mvn()
        x = q.rsample(torch.Size([200]))
        lq = manif.log_q(q.log_prob, x, manif.d, kmax=kmax)
        H = -lq.mean(dim=0).detach().numpy()
        std = lq.std(dim=0).detach().numpy()

        Hmax = -manif.lprior_const
        Hgauss = d / 2 + d / 2 * np.log(2 * np.pi) + d * np.log(sigmas)

        if savefig:
            plt.figure()
            plt.plot([sigmas[0], sigmas[-1]], [Hmax, Hmax], 'g-')
            plt.plot(sigmas[:, 0], Hgauss, 'g-')
            plt.plot(sigmas[:, 0], H, 'b--')
            plt.fill_between(sigmas[:, 0],
                             H - std,
                             H + std,
                             color='b',
                             alpha=0.2)
            plt.xscale('log')
            plt.savefig('test_torus' + str(d) + '.png', dpi=120)
            plt.close()

        assert np.amax(H - std) < Hmax  # adhere to upper bound
        assert np.abs(H[0] - Hgauss[0]) < std[0]  # adhere to lower bound


def test_so3(kmax=5, savefig=False):
    m = 100
    manif = So3(m)
    sigmas = 10**np.linspace(-2, 1, num=m).reshape(m, 1)
    q = mgplvm.rdist.ReLie(manif, m, sigma=torch.tensor(sigmas)).mvn()
    x = q.rsample(torch.Size([200]))
    lq = manif.log_q(q.log_prob, x, manif.d, kmax=kmax)
    H = -lq.mean(dim=0).detach().numpy()
    std = lq.std(dim=0).detach().numpy()

    Hmax = -manif.lprior_const
    Hgauss = 3 / 2 + 3 / 2 * np.log(2 * np.pi) + 3 * np.log(sigmas)

    if savefig:
        plt.figure()
        plt.plot([sigmas[0], sigmas[-1]], [Hmax, Hmax], 'g-')
        plt.plot(sigmas[:, 0], Hgauss, 'g-')
        plt.plot(sigmas[:, 0], H, 'b--')
        plt.fill_between(sigmas[:, 0], H - std, H + std, color='b', alpha=0.2)
        plt.xscale('log')
        plt.savefig('test_so3.png', dpi=120)
        plt.close()

    assert np.amax(H - std) < Hmax  # adhere to upper bound
    assert np.abs(H[0] - Hgauss[0]) < std[0]  # adhere to lower bound


def test_s3(kmax=5, savefig=False):
    m = 100

    manif = S3(m)
    sigmas = 10**np.linspace(-2, 1, num=m).reshape(m, 1)
    q = mgplvm.rdist.ReLie(manif, m, sigma=torch.tensor(sigmas)).mvn()

    x = q.rsample(torch.Size([200]))
    lq = manif.log_q(q.log_prob, x, manif.d, kmax=kmax)
    H = -lq.mean(dim=0).detach().numpy()
    std = lq.std(dim=0).detach().numpy()

    Hmax = -manif.lprior_const
    Hgauss = 3 / 2 + 3 / 2 * np.log(2 * np.pi) + 3 * np.log(sigmas)

    if savefig:
        plt.figure()
        plt.plot([sigmas[0], sigmas[-1]], [Hmax, Hmax], 'g-')
        plt.plot(sigmas[:, 0], Hgauss, 'g-')
        plt.plot(sigmas[:, 0], H, 'b--')
        plt.fill_between(sigmas[:, 0], H - std, H + std, color='b', alpha=0.2)
        plt.xscale('log')
        plt.savefig('test_s3.png', dpi=120)
        plt.close()

    assert np.amax(H - std) < Hmax  # adhere to upper bound
    assert np.abs(H[0] - Hgauss[0]) < std[0]  # adhere to lower bound


if __name__ == "__main__":
    test_euclid()
    test_torus()
    test_so3()
    test_s3()
    print('Tested entropies')
