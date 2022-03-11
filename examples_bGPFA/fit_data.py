"""
In this file, we fit a slightly larger dataset from a primate reaching task.
The dataset is adapted from O'Doherty et al. (2017) and can be downloaded here:
https://drive.google.com/file/d/1UNd8_uWf_jLtEaU0uIxteh2otGfTAKu-/view?usp=sharing
"""

#import the packages we need
import mgplvm as mgp #note that our GPLVM library must be installed first
import torch
import numpy as np
import pickle
import time

### set pytorch device and random seeds ###
device = mgp.utils.get_device() #default is GPU if available
np.random.seed(0)
torch.manual_seed(0)

#### load some example data ####
data = pickle.load(open('Doherty_example.pickled', 'rb'))
binsize = 25 #binsize in ms
timepoints = np.arange(3000, 6000) #subsample 75 seconds of data so things will run somewhat quicker
fit_data = {'Y': data['Y'][..., timepoints], 'locs': data['locs'][timepoints, :], 'targets': data['targets'][timepoints, :], 'binsize': binsize}
Y = fit_data['Y'] # these are the actual recordings and is the input to our model

Y = Y[:, np.mean(Y,axis = (0, 2))/0.025 > 6, :] #subsample highly active neurons so things will run a bit quicker
ntrials, n, T = Y.shape # Y should have shape: [number of trials (here 1) x neurons x time points]
data = torch.tensor(Y).to(device) # put the data on our GPU/CPU
ts = np.arange(Y.shape[-1]) #much easier to work in units of time bins here
fit_ts = torch.tensor(ts)[None, None, :].to(device) # put our time points on GPU/CPU

### set some parameters for fitting ###
ell0 = 200/binsize #initial timescale (in bins) for each dimension. This could be the ~timescale of the behavior of interest (otherwise a few hundred ms is a reasonable default)
rho = 3 #sets the intial scale of each latent (s_d in Jensen & Kao). rho=1 is a natural choice with Gaussian noise; less obvious with non-Gaussian noise but rho=1-5 works well empirically.
max_steps = 2501 #number of training iterations
n_mc = 10 #number of monte carlo samples per iteration
print_every = 100 #how often we print training progress
d_fit = 15 #lets fit up to 15 latent dimensions (in theory this could be up to the number of neurons; should be thought of as an upper bound to how high-dimensional the activity is)

### here are some important parameters to deal with memory limitations! The memory cost is ~O( [T (or batch_size)] * [d_fit^2] * [n] * [n_mc (or batch_mc)] ) ###
batch_mc = None #parameter used to batch gradients across mc samples. 'None' does not batch. Set to smaller number (e.g. 1) in case of memory errors.
batch_size = None #parameter used to batch gradients across time bins. 'None' does not batch. Set to smaller number (e.g. 5000 or 2000) in case of memory errors. Set batch_mc=1 first.


### construct the actual model ###
lik = mgp.likelihoods.NegativeBinomial(n, Y=Y) # we use a negative binomial noise model in this example (recommended for ephys data)
manif = mgp.manifolds.Euclid(T, d_fit) # our latent variables live in a Euclidean space for bGPFA (see Jensen et al. 2020 for alternatives)
var_dist = mgp.rdist.GP_circ(manif, T, ntrials, fit_ts, _scale=1, ell = ell0) # circulant variational GP posterior (c.f. Jensen & Kao et al. 2021)
lprior = mgp.lpriors.Null(manif) # here the prior is defined implicitly in our variational distribution, but if we wanted to fit e.g. Factor analysis this would be a Gaussian prior
mod = mgp.models.Lvgplvm(n, T, d_fit, ntrials, var_dist, lprior, lik, Y = Y, learn_scale = False, ard = True, rel_scale = rho).to(device) #create bGPFA model with ARD


### training will proceed for 2500 iterations (this should take ~30 minutes on GPU) ###
t0 = time.time()
def cb_ard(mod, i, loss):
    """here we construct an (optional) function that helps us keep track of the training"""
    if i % print_every == 0:
        sd = np.log(mod.obs.dim_scale.detach().cpu().numpy().flatten())
        print('\niter:', i, 'time:', str(round(time.time()-t0))+'s', 'scales:', np.round(sd[np.argsort(-sd)], 1))

# helper function to specify training parameters
train_ps = mgp.crossval.training_params(max_steps = max_steps, n_mc = n_mc, lrate = 5e-2, callback = cb_ard, print_every = print_every, batch_size = batch_size, batch_mc = batch_mc)
print('fitting', n, 'neurons and', T, 'time bins for', max_steps, 'iterations')
mod_train = mgp.crossval.train_model(mod, data, train_ps)


#save the model for our separate analysis script
torch.save(mod, 'example_primate_model.pt')
pickle.dump(fit_data, open('fitted_example_data.pickled', 'wb'))



