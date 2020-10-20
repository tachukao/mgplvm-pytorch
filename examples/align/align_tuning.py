import numpy as np
import scipy.stats
import torch


def align_torus_tuning(Y_test, mod_test, Y_target, mod_target, res=200):
    # maximize corelation between all tuning curves
    angles = np.linspace(0, 2 * np.pi, res + 1)[:-1]
    query = torch.tensor(angles, dtype=torch.get_default_dtype())

    data_test = torch.tensor(Y_test, dtype=torch.get_default_dtype())
    ftest, _ = mod_test.predict(data_test, query, niter=50)

    data_target = torch.tensor(Y_target, dtype=torch.get_default_dtype())
    ftarget, _ = mod_target.predict(data_target, query, niter=50)

    maxres = 0
    maxcor = -1
    maxsign = 1
    for sign in [1, -1]:
        for i in range(res):
            newf = np.roll(ftest[:, :, 0], i, axis=1)
            if sign == -1:
                newf = np.flip(newf, axis=1)
            newcor = scipy.stats.pearsonr(ftarget.flatten(), newf.flatten())[0]
            if newcor > maxcor:
                maxcor = newcor
                maxres = i
                maxsign = sign
            if i % 20 == 0:
                print(sign, i, newcor)

    shift = angles[maxres]
    newparam = mod_test.manif.mu + shift
    if maxsign == -1:
        newparam = 2 * np.pi - newparam
    newparam = newparam % (2 * np.pi)
    device = mod_test.manif.mu.device
    mod_test.manif.mu = torch.nn.Parameter(newparam).to(device)

    return mod_test
