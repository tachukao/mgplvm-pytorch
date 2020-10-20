import numpy as np
import csv
from scipy.stats import circmean, circvar, binned_statistic


def load_channel(mouse='28', session='140313', channel='1', freq=20000):
    '''
    freq is frequency in Hz
    '''
    mouse = 'Mouse' + mouse + '-' + session
    basedata = mouse + '/'
    data = []
    with open(basedata + mouse + '.clu.' + channel) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        clusts = np.array([val[0] for val in reader])
    nclusts = int(clusts[0])
    clusts = clusts[1:].astype(int)
    with open(basedata + mouse + '.res.' + channel) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        spiketimes = np.array([val[0] for val in reader])

    for cluster in range(nclusts):
        cluster_times = spiketimes[clusts == cluster]
        data.append(cluster_times / freq)

    return data


def load_channels(mouse='28', session='140313', channels=range(1, 9)):
    data = []
    for channel in channels:
        data = data + load_channel(
            mouse=mouse, session=session, channel=str(channel))
    return data


def load_states(mouse='28', session='140313'):
    '''
    wake, REM and SWS
    '''

    mouse = 'Mouse' + mouse + '-' + session
    basedata = mouse + '/'
    data = []
    for state in ['Wake', 'REM', 'SWS']:
        with open(basedata + mouse + '.states.' + state) as csvfile:
            reader = csv.reader(csvfile,
                                quoting=csv.QUOTE_NONNUMERIC,
                                delimiter='\t')
            data.append(np.array([val for val in reader]))

    return data[0], data[1], data[2]


def load_angs(mouse='28', session='140313'):

    mouse = 'Mouse' + mouse + '-' + session
    with open(mouse + '.ang') as csvfile:
        reader = csv.reader(csvfile,
                            delimiter='\t',
                            quoting=csv.QUOTE_NONNUMERIC)
        data = np.array([val for val in reader])

    inds = (data[:, 1] >= 0)  # only take timepoints with data

    times = data[inds, 0]
    angs = data[inds, 1]

    return times, angs


def bin_data(data, binsize=1):
    maxval = np.amax([dat[-1] for dat in data])
    minval = 0
    bins = np.arange(minval, np.ceil(maxval), binsize)
    bindata = [np.histogram(dat, bins=bins)[0] for dat in data]
    return bindata, bins


def bin_angs(times, angs, bins):
    '''
    can also do scipy.circvar if we want to check that the variability is reasonable
    '''

    def fmean(samples):
        return circmean(samples, nan_policy='propagate')

    binangs = binned_statistic(angs, times, statistic=fmean, bins=bins)[0]

    def fvar(samples):
        return circvar(samples, nan_policy='propagate')

    binvars = binned_statistic(angs, times, statistic=fvar, bins=bins)[0]

    return binangs, np.sqrt(binvars)  # return mean and std


def bin_states(states, bins):
    data = []
    for state in states:
        binstates = np.zeros(len(bins) - 1)
        for row in state:  # add data corresponding to each row
            cond1 = (bins[:-1] > row[0])
            cond2 = (bins[1:] < row[1])
            binstates[(cond1 & cond2)] = 1
        data.append(binstates.astype(bool))

    return data[0], data[1], data[2]
