
'''
follow Blair et al. 2007, J neurosci
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle
np.random.seed(14052310)

def calc_K(ts, sig = 10, l = 2):
    '''compute kernel for generating trajectories'''
    rep_ts = ts.reshape(-1, 1).repeat(len(ts), 1)
    dts = rep_ts - rep_ts.T

    return sig**2 * np.exp(-dts**2 / (2*l**2))


class grid_cell():
    
    def __init__(self, c, module):
        self.c = np.array(c)
        self.module = module
        return
    
    
    def calc_G(self, ws, X):
        '''compute grid activity (Blair et al.)'''
        dX = X - self.c.reshape(1,2)
        w_dX = ws @ dX.T

        G = np.cos( w_dX ).sum(axis = 0)
        G = np.exp(0.3*(G+3/2))-1

        return G
    
    def comp_tuning(self, xmin, xmax, nx):
        xs = np.linspace(xmin, xmax, nx)
        Xs, Ys = np.meshgrid(*[xs, xs])
        X = np.array([Xs.flatten(), Ys.flatten()]).T
        G = self.calc_G(self.module.ws, X)
        
        return G.reshape(nx, nx)
        

class grid_module():
    
    def __init__(self, lambd, theta, n_c = 7):
        '''generate a single grid module with fixed spacing (lambda) and orientation(theta)'''
        #generate grid vectors
        theta_ws = [(theta+dtheta)*np.pi/180 for dtheta in [-30, 30, 90]]
        self.ws = np.array([
                            [np.cos(wtheta), np.sin(wtheta)] for wtheta in theta_ws
                        ]) * 4*np.pi/np.sqrt(3)/lambd
        
        self.cells = []
        for n1 in range(n_c):
            for n2 in range(n_c): 
                self.add_cell(np.array([n1, n2])*2*np.pi/n_c)
        
        self.theta = theta
        self.lambd = lambd
        self.n_c = n_c
        
        return
    
    def add_cell(self, c):
        '''add a cell with phase c'''
        self.cells.append(grid_cell(c, self))
    
    def calc_G(self, X):
        '''compute activity for all cells in this module at locations X'''
        G = []
        for cell in self.cells:
            G.append(cell.calc_G(self.ws, X))
            
        return np.array(G)
    
class grid_pop():
    
    def __init__(self, n_modules = 3, lambda0 = 2, lambda_ratio = 1.618, n_c = 7):
        '''initialize a population of grid cells
        n_modules: number of grid modules
        lambda0: spacing in module with smallest spacing
        lambda_ratio: ratio of spacings between modules'''
        
        self.lambda0, self.lambda_ratio, self.n_c = lambda0, lambda_ratio, n_c
        
        self.modules = []
        for n in range(n_modules):
            self.add_module()
        
        return
    
    def add_module(self, lambd = None, theta = 30, n_c = None):
        '''add a module to the population with n_c**2 neurons'''
        if lambd is None:
            lambd = self.lambda0 * self.lambda_ratio**(len(self.modules)+1)
        n_c = self.n_c if n_c == None else n_c
        self.modules.append(grid_module(lambd = lambd, theta = theta, n_c = n_c))
    
    def calc_G(self, X = None):
        '''Compute all cell activities for a given trajectory X'''
        if X is None: X = self.X
        G = []
        for module in self.modules:
            G.append(module.calc_G(X))
            
        return np.concatenate(G, axis = 0)
    
    def gen_trajec_GP(self, tmax = 100, T = 500, l = 2, sig = 10):
        '''generate behavioral data drawn from a GP with ell = l.
        T: number of timepoints to generate between 0 and T'''
        self.T = T #timepoints
        self.ts = np.linspace(0, tmax, self.T) #100s at 10Hz

        K = calc_K(self.ts, sig = sig, l = l) #Kernel
        L = np.linalg.cholesky(K + 1e-6 * np.eye(self.T))

        us = np.random.normal(size = (self.T, 2))
        X = L @ us #sample x, y from GP

        self.X = X
        return X
    
    def comp_tuning(self, xmin, xmax, nx):
        G = []
        for module in self.modules:
            for cell in module.cells:
                G.append(cell.comp_tuning(xmin, xmax, nx))
        return G


        
'''

#%% generate neural activity


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
'''

tests = False #make some plots etc.
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

