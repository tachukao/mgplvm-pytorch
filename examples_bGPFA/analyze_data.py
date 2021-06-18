"""
In this notebook we run a few simple analyses on the model we trained using fit_data.py
"""

### import the packages needed ###
import mgplvm as mgp #note that our GPLVM library must be installed first
import torch
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression
from scipy.interpolate import CubicSpline
plt.rcParams['font.size'] = 20
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
device = mgp.utils.get_device() #default is GPU if available


### load the trained model and the data we fitted ###
mod = torch.load('example_primate_model.pt')
fit_data = pickle.load(open('fitted_example_data.pickled', 'rb'))

### we start by plotting 'informative' and 'discarded' dimensions ###
print('plotting informative and discarded dimensions')
dim_scales = mod.obs.dim_scale.detach().cpu().numpy().flatten() #prior scales (s_d)
dim_scales = np.log(dim_scales) #take the log of the prior scales
nus = np.sqrt(np.mean(mod.lat_dist.nu.detach().cpu().numpy()**2, axis = (0, -1))) #magnitude of the variational mean
plt.figure()
plt.scatter(dim_scales, nus, c = 'k', marker = 'x', s = 30) #top right corner are informative, lower left discarded
plt.xlabel(r'$\log \, s_d$')
plt.ylabel(r'$||\nu_d||^2_2$', labelpad = 5)
plt.savefig('figures/primate_dimensions.png', bbox_inches = 'tight')
plt.close()


### plot the inferred latent trajectories
print('plotting latent trajectories')
X = mod.lat_dist.lat_mu.detach().cpu().numpy()[0, ...] #extract inferred latents ('mu' has shape (ntrials x T x d_fit))
X = X[..., np.argsort(-dim_scales)[:2]] #only consider the two most informative dimensions (c.f. Jensen & Kao)
tplot = np.arange(100, 200) #let's only plot a shorter period (here 2.s) so it doesn't get too cluttered

#fit FA for comparison
fa = FactorAnalysis(2)
Y = fit_data['Y']
Xfa = fa.fit_transform(np.sqrt(Y[0, ...].T)) #sqrt the counts for variance normalization (c.f. Yu et al. 2009)

fig, axs = plt.subplots(1, 2, figsize = (10, 5))
axs[0].plot(X[tplot, 0], X[tplot, 1], 'k-') # plot bGPFA latents
axs[1].plot(Xfa[tplot, 0], Xfa[tplot, 1], 'k-') # plot FA latents
for ax in axs:
    ax.set_xlabel('latent dim 1')
    ax.set_ylabel('latent dim 2')
    ax.set_xticks([])
    ax.set_yticks([])
axs[0].set_title('Bayesian GPFA')
axs[1].set_title('Factor analysis')
plt.savefig('figures/primate_latents.png', bbox_inches = 'tight')
plt.close()


### finally let's do a simple decoding analysis ###
print('running decoding analysis')
Ypreds = [] #decode from the inferred firing rates (this is a non-linear decoder from latents)
query = mod.lat_dist.lat_mu.detach().transpose(-1, -2).to(device)  #(ntrial, d_fit, T)
for i in range(100): #loop over mc samples to avoid memory issues
    Ypred = mod.svgp.sample(query, n_mc=10, noise=False)
    Ypred = Ypred.detach().mean(0).cpu().numpy()  #(ntrial x n x T)
    Ypreds.append(Ypred)
Ypred = np.mean(np.array(Ypreds), axis = (0,1)).T #T x n

locs = fit_data['locs'] #hand positions

delays = np.linspace(-150, 300, 50) #consider different behavioral delays
performance = np.zeros(len(delays)) #model performance
for idelay, delay in enumerate(delays):
    ts = np.arange(Ypred.shape[0])*fit_data['binsize'] #measured in ms
    cs = CubicSpline(ts, locs) #fit cubic spline to behavior
    vels = cs(ts+delay, 1) #velocity at time+delay
    regs = [LinearRegression().fit(Ypred, vels[:, i]) for i in range(2)] #fit x and y vel
    scores = [regs[i].score(Ypred, vels[:, i]) for i in range(2)] #score x and y vel
    performance[idelay] = np.mean(scores) #save performance
print('plotting decoding')
plt.figure()
plt.plot(delays, performance, 'k-')
plt.xlim(delays[0], delays[-1])
plt.xlabel('delay')
plt.ylabel('kinematic decoding')
plt.savefig('figures/primate_decoding.png', bbox_inches = 'tight')
plt.close()



