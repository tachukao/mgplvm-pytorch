
'''
follow Blair et al. 2007, J neurosci
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle
np.random.seed(14052310)

def calc_K(ts, sig = 10, l = 0.5):
    '''compute kernel for generating trajectories'''
    rep_ts = ts.reshape(-1, 1).repeat(T, 1)
    dts = rep_ts - rep_ts.T

    return sig**2 * np.exp(-dts**2 / (2*l**2))

def calc_G(theta, lambd, c, R):
    '''compute grid activity (Blair et al.)'''
    theta_ws = [theta+dtheta*np.pi/180 for dtheta in [-30, 30, 90]]
    ws = np.array([
        [np.cos(wtheta), np.sin(wtheta)] for wtheta in theta_ws
    ]) * 4*np.pi/np.sqrt(3)/lambd

    dR = R - c.reshape(1,2)
    w_dR = ws @ dR.T

    G = np.cos( w_dR ).sum(axis = 0)
    G = np.exp(0.3*(G+3/2))-1

    return G


#%% generate behavioral data from a GP
T = 500 #timepoints
ts = np.linspace(0, 100, T) #100s at 10Hz

K = calc_K(ts) #Kernel
L = np.linalg.cholesky(K + 1e-6 * np.eye(T))

us = np.random.normal(size = (T, 2))
X = L @ us #sample x, y from GP

#%% generate neural activity

n_modules = 3 #number of modules
grid_ratio = 1.618 #ratio between grid size of subsequent modules
lambda_init = 2 #smallest grid ratio
lambds = [lambda_init*grid_ratio**n for n in range(n_modules)]
theta = 30*np.pi/180 #orientation is constant

Y = [] #firing rates
all_lambds = [] #grid sizes
all_cs = [] #grid phases
for lambd in lambds:

    #tile the module with 7x7 phases
    c1s = np.linspace(0, lambd, 8)[:-1]
    c1s, c2s = np.meshgrid(*[c1s, c1s])
    cs = np.array([c1s.flatten(), c2s.flatten()]).T

    for c in cs:
        #generate grid cell activity for each phase and store data
        Y.append(calc_G(theta, lambd, c, X))
        all_lambds.append(lambd)
        all_cs.append(cs)

#plot heatmap of firing rates
Y = np.array(Y)
plt.figure()
plt.imshow(Y, cmap = 'Greys', aspect = 'auto')
plt.show()
plt.close()

#save data for future use
data = {'Y': Y,
        'lambdas': all_lambds,
        'cs': all_cs}
pickle.dump(data, open('grid_data.pickled', 'wb'))


tests = True #make some plots etc.
if tests:

    #plot behavior
    plt.figure()
    plt.plot(ts, X[:, 0])
    plt.plot(ts, X[:, 1])
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(X[:,0], X[:, 1])
    plt.show()
    plt.close()

    #%% generate grid cells

    lambdas = np.array([lambda_init, lambda_init, lambda_init, lambda_init,
                lambda_init*grid_ratio, lambda_init*grid_ratio**2])
    cs = np.array([
        [0, 0],
        [0, 0.5],
        [0.5, 0],
        [0.5, 0.5],
        [0, 0],
        [1, 1]
    ])
    N = len(lambdas)
    thetas = np.array([theta for i in range(N)])

    ##% tuning grid
    xs = np.linspace(0, 10, 50)
    ys = np.linspace(0, 10, 50)
    Xs, Ys = np.meshgrid(*[xs, ys])
    Rs = np.array([Xs.flatten(), Ys.flatten()]).T


    for i in range(N):
        theta, lambd, c = thetas[i], lambdas[i], cs[i]
        G = calc_G(theta, lambd, c, Rs)
        plt.figure()
        plt.imshow(G.reshape(len(xs), len(ys)), cmap = 'Greys')
        plt.xticks([-0.5, len(xs)-0.5], [0, xs[-1]])
        plt.yticks([-0.5, len(ys)-0.5], [0, ys[-1]])
        plt.show()
        plt.close()

