import numpy as np
import os
import pickle
from load_data import load_channels, load_angs, bin_data, bin_angs, load_states, bin_states

binsize = 0.5  # resolution in seconds

# from metadata; which channels recorded from ADn
channeladn = {'28_140313': (8, 11)}
# from metadata; which channels recorded from postsubiculum
channelpos = {'28_140313': (1, 7)}

channeldict = channeladn  # we're analyzing ADn
key = '28_140313'
mouse, session = key.split('_')
channels = range(channeldict[mouse + '_' + session][0],
                 channeldict[mouse + '_' + session][1] + 1)

print('\nextracting, binning and analyzing data for mouse', mouse, 'session',
      session, 'electrodes:', list(channels))

print('loading spike data')
data = load_channels(mouse=mouse, session=session, channels=channels)

print('loading behavioral data')
times, angs = load_angs(mouse=mouse, session=session)
wake, rem, _ = load_states(mouse=mouse, session=session)  # ignore sws for now

print('removing spikes with mean(r) < 0.25Hz')
tmax = times[-1]
nmin = tmax * 0.25
data = [dat for dat in data if len(dat) > nmin]

print('binning data')
# bin the neural activity
bindata, bins = bin_data(data, binsize=binsize)
binangs, binang_sds = bin_angs(angs, times, bins)  # bin HD
bin_wake, bin_sleep, _ = bin_states([wake, rem, []], bins)  # bin states

print('collecting data')
# collect firing rates and head directions for wake and sleep
bin_wake[np.isnan(binangs)] = False
bin_sleep[np.isnan(binangs)] = False
ts_wake, ts_sleep = bins[1:][bin_wake], bins[1:][bin_sleep]
zs_wake, zs_sleep = binangs[bin_wake], binangs[bin_sleep]

Y_wake, Y_sleep = [
    np.array([data[bins]
              for data in bindata])
    for bins in [bin_wake, bin_sleep]
]

# match the amount of data for wake and sleep
tmax = np.amin([np.sum(bin_wake), np.sum(bin_sleep)])
ts_wake, ts_sleep = ts_wake[:tmax] - ts_wake[0], ts_sleep[:tmax] - ts_sleep[0]
zs_wake, zs_sleep = zs_wake[:tmax], zs_sleep[:tmax]
Y_wake, Y_sleep = Y_wake[:, :tmax], Y_sleep[:, :tmax]

print('saving data to pickled file')
output_data = {
    'Y_wake': Y_wake,
    'Y_sleep': Y_sleep,
    'hd_wake': zs_wake,
    'hd_sleep': zs_sleep,
    'ts_wake': ts_wake,
    'ts_sleep': ts_sleep
}

pickle.dump(output_data, open('binned_data.pickled', 'wb'))
