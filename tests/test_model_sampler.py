import numpy as np
import torch
from torch import optim
import mgplvm as mgp
import matplotlib.pyplot as plt

torch.manual_seed(1)
np.random.seed(0)

torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def test_sampling():
    """
    test that we can sample from a trained latent variable model
    """
    d = 3  # dims of latent space
    n = 8  # number of neurons
    m = 10  # number of conditions / time points
    n_z = 5  # number of inducing points
    n_mc = 16
    n_samples = 2  # number of samples
    gen = mgp.syndata.Gen(mgp.syndata.Euclid(d),
                          n,
                          m,
                          variability=0.25,
                          n_samples=n_samples)
    Y = gen.gen_data()
    data = torch.tensor(Y, device=device, dtype=torch.get_default_dtype())

    for i, lik in enumerate([
            mgp.likelihoods.Gaussian(n),
            mgp.likelihoods.Poisson(n),
            mgp.likelihoods.NegativeBinomial(n)
    ]):

        # specify manifold, kernel and rdist
        manif = mgp.manifolds.Euclid(m, d)
        lat_dist = mgp.rdist.ReLie(manif, m, n_samples, diagonal=False)
        kernel = mgp.kernels.QuadExp(n, manif.distance)
        lprior = mgp.lpriors.Uniform(manif)
        z = manif.inducing_points(n, n_z)
        mod = mgp.models.SvgpLvm(n,
                                 m,
                                 n_samples,
                                 z,
                                 kernel,
                                 lik,
                                 lat_dist,
                                 lprior,
                                 whiten=True).to(device)

        # train model
        mgp.optimisers.svgp.fit(data,
                                mod,
                                optimizer=optim.Adam,
                                max_steps=5,
                                burnin=5 / 2E-2,
                                n_mc=16,
                                lrate=5E-2,
                                print_every=1000)

        #sample from the model
        query = mod.lat_dist.prms[0].detach().transpose(
            -1, -2)  #(n_samples, m, d) -> (n_samples, d, m)
        y_samps = mod.svgp.sample(query, n_mc=n_mc).cpu().numpy()
        print('\n', i, lik.name, y_samps.shape)
        assert y_samps.shape == (n_mc, n_samples, n, m)


if __name__ == '__main__':
    test_sampling()
