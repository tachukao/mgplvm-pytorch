import torch
from torch.utils.data import Dataset


class NeuralDataLoader:
    def __init__(self,
                 data,
                 batch_size=None,
                 sample_size=None,
                 batch_pool=None,
                 sample_pool=None):
        n_samples, _, m = data.shape
        self.n_samples = n_samples
        self.m = m
        self.batch_pool = batch_pool
        self.sample_pool = sample_pool
        self.batch_pool_size = m if batch_pool is None else len(batch_pool)
        self.sample_pool_size = n_samples if sample_pool is None else len(
            sample_pool)
        self.batch_size = self.batch_pool_size if batch_size is None else batch_size
        self.sample_size = self.sample_pool_size if sample_size is None else sample_size
        if sample_pool is not None:
            data = data[sample_pool]
        if batch_pool is not None:
            data = data[:, :, batch_pool]
        self.data = data
        if self.batch_size > self.batch_pool_size:
            raise Exception(
                "batch size greater than number of conditions in pool")
        if self.sample_size > self.sample_pool_size:
            raise Exception(
                "sample size greater than number of samples in pool")

    def get_next(self):
        i0 = self.i
        i1 = i0 + self.sample_size
        k0 = self.k
        k1 = k0 + self.batch_size
        if i1 > self.sample_pool_size:
            i1 = self.sample_pool_size
        if k1 > self.batch_pool_size:
            k1 = self.batch_pool_size
        batch = self.data[i0:i1][:, :, k0:k1]
        self.k = k1
        batch_idxs = list(range(k0, k1))
        sample_idxs = list(range(i0, i1))
        return sample_idxs, batch_idxs, batch

    def __iter__(self):
        self.i = 0
        self.k = 0
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
