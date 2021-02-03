import numpy as np
import mgplvm as mgp


def test_sampler():
    n_samples = 10
    m = 20
    n = 10
    sample_size = 3
    batch_size = 7
    Y = np.arange(n_samples * m * n).reshape(n_samples, n, m)
    for shuffle_batch in [True, False]:
        for shuffle_sample in [True, False]:
            dataloader = mgp.optimisers.data.BatchDataLoader(
                Y, sample_size=sample_size, batch_size=batch_size)

            k = int(np.ceil(m / batch_size)) * int(
                np.ceil(n_samples / sample_size))
            assert (k == len(list(dataloader)))
            # loop through the dataloader 3 times
            # to see that it works even with in-place shuffling
            for _ in range(3):
                for sample_idxs, batch_idxs, batch in dataloader:
                    assert (batch.shape[1] == n)
                    assert (batch.shape[0] <= sample_size)
                    assert (batch.shape[2] <= batch_size)
                    assert np.alltrue(Y[sample_idxs][:, :,
                                                     batch_idxs] == batch)


def test_sampler_pool():
    n_samples = 10
    m = 20
    n = 10
    sample_size = 3
    batch_size = 7
    batch_pool = list(range(2, m // 2 + 2))
    sample_pool = list(range(2, n_samples // 2 + 2))
    Y = np.arange(n_samples * m * n).reshape(n_samples, n, m)
    for shuffle_batch in [True, False]:
        for shuffle_sample in [True, False]:
            dataloader = mgp.optimisers.data.BatchDataLoader(
                Y,
                batch_pool=batch_pool,
                sample_pool=sample_pool,
                sample_size=sample_size,
                batch_size=batch_size,
                shuffle_batch=shuffle_batch,
                shuffle_sample=shuffle_sample)
            k = int(np.ceil(len(batch_pool) / batch_size)) * int(
                np.ceil(len(sample_pool) / sample_size))
            assert (k == len(list(dataloader)))
            # loop through the dataloader 3 times
            # to see that it works even with in-place shuffling
            for _ in range(2):
                for sample_idxs, batch_idxs, batch in dataloader:
                    assert (batch.shape[1] == n)
                    assert (batch.shape[0] <= sample_size)
                    assert (batch.shape[2] <= batch_size)
                    assert np.alltrue(Y[sample_idxs][:, :,
                                                     batch_idxs] == batch)


if __name__ == "__main__":
    test_sampler()
    test_sampler_pool()
