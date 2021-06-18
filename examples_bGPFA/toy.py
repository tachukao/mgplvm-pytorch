"""
In this file, we generate a small synthetic dataset and fit bGPFA with a negative binomial noise model
"""

### import the packages needed ###
import mgplvm as mgp #note that our GPLVM library must be installed first
import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
plt.rcParams['font.size'] = 20
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

### set pytorch device and random seeds ###
device = mgp.utils.get_device() #default is GPU if available
np.random.seed(0)
torch.manual_seed(0)
def detach(tensor):
    """turns a pytorch parameter tensor into a numpy array (useful for plotting)"""
    return tensor.detach().cpu().numpy()

#### generate synthetic latents and tuning curves ####
n, T = 35, 180 #35 neurons, 180 time points
ts = np.arange(T) # time points
dts_2 = (ts[:, None] - ts[None, :])**2 # compute dts for the kernel
ell = 20 # effective length scale
d_true = 2 # 2 ground truth dimensions
K = np.exp(-dts_2/(2*ell**2)) # TxT covariance matrix
L = np.linalg.cholesky(K + np.eye(T)*1e-6) # TxT cholesky factor
xs = (L @ np.random.normal(0, 1, (T, d_true))).T # DxT true latent states
C = np.random.normal(0, 1, (n, d_true))*0.5 # factor matrix
F = C @ xs # n x T true de-noised activity

#### draw noise from NegBinomial model ####
c_nb = -1.50 #scale factor for reasonable magnitude of activity
p_nb = np.exp(F+c_nb)/(1+np.exp(F+c_nb)) #probability of failure (c.f. Jensen & Kao et al.)
r_nb = np.random.uniform(1, 10, n) # number of failures (overdispersion paramer; c.f. Jensen & Kao et al.)
#numpy defines it's negative binomial distribution in terms of #successes so we substitute 1 -> 1-p
YNB = np.random.negative_binomial(r_nb, 1-p_nb.T).astype(float).T

t0 = time.time()
def cb_ard(mod, i, loss):
    """here we construct an (optional) function that helps us keep track of the training"""
    if i % 100 == 0:
        sd = np.log(detach(mod.obs.dim_scale).flatten())
        print('\niter:', i, 'time:', str(round(time.time()-t0))+'s', 'scales:', np.round(sd[np.argsort(-sd)], 1))
    return False

### set up the model ###

Y = YNB[None, ...] # we add a trial dimension which isn't actually used here but could be if we had multiple trials
data = torch.tensor(Y).to(device) # put the data on our GPU/CPU
fit_ts = torch.tensor(ts)[None, None, :].to(device) # put our time points on GPU/CPU

d_fit = 10 #lets fit up to 10 latent dimensions (in theory this could just be the number of neurons if we don't have any prior to start from)
lik = mgp.likelihoods.NegativeBinomial(n, Y=Y) # we use a negative binomial noise model in this example
manif = mgp.manifolds.Euclid(T, d_fit) # our latent variables live in a Euclidean space (c.f. Jensen et al. 2020)
lprior = mgp.lpriors.Null(manif) # here the prior is defined implicitly in our variational distribution, but if we wanted to fit e.g. Factor analysis this would be a Gaussian prior
var_dist = mgp.rdist.GP_circ(manif, T, 1, fit_ts, _scale=1, ell = 20*0.8) # circulant variational posterior (c.f. Jensen & Kao et al.)
mod = mgp.models.Lvgplvm(n, T, d_fit, 1, var_dist, lprior, lik, Y = Y, learn_scale = False, ard = True, rel_scale = 0.1).to(device) #create ARD model

train_ps = mgp.crossval.training_params(max_steps = 2001, n_mc = 10, burnin = 50, lrate = 5e-2, callback = cb_ard, print_every=100) # helper function to specify training parameters

### training will proceed for 2000 iterations (this should take 1-2 minutes on GPU) ###
mod_train = mgp.crossval.train_model(mod, data, train_ps)



#### run a few analyses ####

### plot the activity we generated ###
plt.figure(figsize = (12, 6))
plt.imshow(YNB, cmap = 'Greys', aspect = 'auto')
plt.xlabel('time')
plt.ylabel('neuron')
plt.title('Raw activity', fontsize = 25)
plt.xticks([])
plt.yticks([])
plt.savefig('figures/toy_raster.png', bbox_inches = 'tight')
plt.close()

### identify 'discarded' dimensions ###
dim_scales = detach(mod.obs.dim_scale).flatten() #prior scales (s_d)
dim_scales = np.log(dim_scales) #take the log of the prior scales
nus = np.sqrt(np.mean(detach(mod.lat_dist.nu)**2, axis = (0, -1))) #magnitude of the variational mean
plt.figure()
plt.scatter(dim_scales, nus, c = 'k', marker = 'x', s = 30) #top right corner are informative, lower left discarded
plt.xlabel(r'$\log \, s_d$')
plt.ylabel(r'$||\nu_d||^2_2$', labelpad = 5)
plt.savefig('figures/toy_dimensions.png', bbox_inches = 'tight')
plt.close()

### plot true and inferred latent trajectories ###
lats = detach(mod.lat_dist.lat_mu)[0, ...] #extract inferred latents
lats = lats[..., np.argsort(-dim_scales)[:2]] #only consider the two most informative dimensions (c.f. Jensen & Kao et al.)
true_lats = xs.T # T x 2

# here we will align the latent trajectories using linear regression since our model is only specified up to a rotation and scaling
lats = lats - np.mean(lats, axis = 0, keepdims = True) #zero-center
true_lats = true_lats - np.mean(true_lats, axis = 0, keepdims = True) #zero-center
T = np.linalg.inv(lats.T @ lats) @ lats.T @ true_lats #regress onto ground truth latents (xs)
lats = lats @ T  #aligned values

plt.figure() #plot latent trajectories
plt.plot(true_lats[:, 0], true_lats[:, 1], 'k-') #plot ground truth
plt.plot(lats[:, 0], lats[:, 1], 'b-') #plot inferred
plt.xlabel('latent dim 1')
plt.ylabel('latent dim 2')
plt.legend(['true', 'inferred'], frameon = False)
plt.savefig('figures/toy_latents.png', bbox_inches = 'tight')
plt.close()

### plot true and inferred overdispersion parameters (c.f. Jensen & Kao et al.) ###
r_inf = detach(mod.obs.likelihood.prms[0]) #inferred overdispersion parameter
plt.figure()
plt.scatter(np.log(r_nb), np.log(r_inf), c = 'k', s = 20) #plot true vs inferred
id_line = np.log(np.array([np.amin(r_nb), np.amax(r_nb)])) #plot identity line for reference
plt.plot(id_line, id_line, 'k-')
plt.xlabel(r'$\log \, \kappa_{true}$')
plt.ylabel(r'$\log \, \kappa_{inf}$')
plt.savefig('figures/toy_kappas.png', bbox_inches = 'tight')
plt.close()

