import numpy as np
import torch
from torch import optim
import mgplvm
from mgplvm import kernels, rdist, models, training
from mgplvm.manifolds import Torus, Euclid, So3
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
d = 1  # dims of latent space
n = 50  # number of neurons
m = 100  # number of conditions / time points
n_z = 15  # number of inducing points
n_samples = 1  # number of samples
gen = mgplvm.syndata.Gen(syndata.Torus(d), n, m, variability=0.25)
sig0 = 1.5
l = 0.4
gen.set_param('l', l)
Y = gen.gen_data()
print('mean activity:', np.mean(Y))
Y = Y + np.random.normal(size=Y.shape) * np.mean(Y) / 3
# specify manifold, kernel and rdist
manif = Torus(m, d)
ref_dist = mgplvm.rdist.MVN(m, d, sigma=sig0)
# initialize signal variance
alpha = np.mean(np.std(Y, axis=1), axis=1)
kernel = kernels.QuadExp(n, manif.distance, alpha=alpha)
# generate model
sigma = np.mean(np.std(Y, axis=1), axis=1)  # initialize noise
likelihoods = mgplvm.likelihoods.Gaussian(n, variance=np.square(sigma))
mod = models.Svgp(manif, n, m, n_z, kernel, likelihoods, ref_dist,
                  whiten=True).to(device)


def callback(model, i):
    if i % 100 == 0:
        gs_true = gen.gs[0]
        gprefs_true = gen.gprefs[0]
        g_mus = mod.manif.prms.data.cpu().numpy()[:]
        plt.figure()
        plt.plot(gs_true[:, 0], g_mus[:, 0], "ko")
        plt.title('i = ' + str(i))
        plt.savefig("yo")
        plt.close()


# train model
trained_model = training.svgp(Y,
                              mod,
                              device,
                              optimizer=optim.Adam,
                              outdir='none',
                              max_steps=10000,
                              burnin=5 / 2E-2,
                              n_mc=64,
                              lrate=2E-2,
                              print_every=20,
                              callback=callback)
