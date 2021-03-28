import torch
import numpy as np
from torch.utils.data import Dataset


class DataLoader:

    def __init__(self, data):
        n_samples, n, m = data.shape
        self.n = n
        self.n_samples = n_samples
        self.m = m
        self.batch_pool_size = m
        self.data = data

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i == 0:
            self.i += 1
            return (None, None, self.data)
        else:
            raise StopIteration


class BatchDataLoader(DataLoader):

    def __init__(self,
                 data,
                 batch_size=None,
                 sample_size=None,
                 batch_pool=None,
                 sample_pool=None,
                 shuffle_batch=False,
                 shuffle_sample=False,
                 overlap=0):
        super(BatchDataLoader, self).__init__(data)
        m = self.m
        n_samples = self.n_samples
        self.overlap = overlap
        self.shuffle_batch = shuffle_batch
        self.shuffle_sample = shuffle_sample
        self.batch_pool = list(range(m)) if batch_pool is None else batch_pool
        self.sample_pool = list(
            range(n_samples)) if sample_pool is None else sample_pool
        self.batch_pool_size = len(self.batch_pool)
        self.sample_pool_size = len(self.sample_pool)
        self.batch_size = self.batch_pool_size if batch_size is None else batch_size
        self.sample_size = self.sample_pool_size if sample_size is None else sample_size
        if sample_pool is not None:
            self.data = self.data[sample_pool]
        if batch_pool is not None:
            self.data = self.data[:, :, batch_pool]
        if self.batch_size > self.batch_pool_size:
            raise Exception(
                "batch size greater than number of conditions in pool")
        if self.sample_size > self.sample_pool_size:
            raise Exception(
                "sample size greater than number of samples in pool")

    def __iter__(self):
        self.i = 0
        self.k = 0
        if self.shuffle_sample:
            sample_shuffle_idxs = list(range(self.sample_pool_size))
            np.random.shuffle(sample_shuffle_idxs)
            self.sample_pool = [
                self.sample_pool[i] for i in sample_shuffle_idxs
            ]
            self.data = self.data[sample_shuffle_idxs]
        if self.shuffle_batch:
            batch_shuffle_idxs = list(range(self.batch_pool_size))
            np.random.shuffle(batch_shuffle_idxs)
            self.batch_pool = [self.batch_pool[i] for i in batch_shuffle_idxs]
            self.data = self.data[:, :, batch_shuffle_idxs]
        return self

    def __next__(self):
        if self.i >= self.sample_pool_size:
            raise StopIteration
        else:
            if self.k >= self.batch_pool_size:
                self.k = 0
                self.i += self.sample_size
                if self.i >= self.sample_pool_size:
                    raise StopIteration
                else:
                    return self.get_next()
            else:
                return self.get_next()

    def get_next(self):
        i0 = self.i
        i1 = i0 + self.sample_size
        k0 = self.k - self.overlap * (self.k > 0)
        k1 = k0 + self.batch_size
        if i1 > self.sample_pool_size:
            i1 = self.sample_pool_size
        if k1 > self.batch_pool_size:
            k1 = self.batch_pool_size
        batch = self.data[i0:i1][:, :, k0:k1]
        self.k = k1
        batch_idxs = list(range(k0, k1))
        batch_idxs = [self.batch_pool[i] for i in batch_idxs]
        sample_idxs = list(range(i0, i1))
        sample_idxs = [self.sample_pool[i] for i in sample_idxs]
        return sample_idxs, batch_idxs, batch
