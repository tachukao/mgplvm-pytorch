import numpy as np
import mgplvm as mgp


def test_sampler():
    n_samples = 10
    m = 20
    n = 10
    sample_size = 3
    batch_size = 7
    Y = np.arange(n_samples * m * n).reshape(n_samples, n, m)
    dataloader = mgp.optimisers.data.NeuralDataLoader(Y,
                                                      sample_size=sample_size,
                                                      batch_size=batch_size)

    k = int(np.ceil(m / batch_size)) * int(np.ceil(n_samples / sample_size))
    assert (k == len(list(dataloader)))
    for sample_idxs, batch_idxs, batch in dataloader:
        assert (batch.shape[1] == n)
        assert (batch.shape[0] <= sample_size)
        assert (batch.shape[2] <= batch_size)
        assert np.alltrue(Y[sample_idxs][:, :, batch_idxs] == batch)


def test_sampler_pool():
    n_samples = 10
    m = 20
    n = 10
    sample_size = 3
    batch_size = 7
    batch_pool = list(range(0, m // 2))
    sample_pool = list(range(0, n_samples // 2))
    Y = np.arange(n_samples * m * n).reshape(n_samples, n, m)
    dataloader = mgp.optimisers.data.NeuralDataLoader(Y,
                                                      batch_pool=batch_pool,
                                                      sample_pool=sample_pool,
                                                      sample_size=sample_size,
                                                      batch_size=batch_size)

    k = int(np.ceil(len(batch_pool) / batch_size)) * int(
        np.ceil(len(sample_pool) / sample_size))
    assert (k == len(list(dataloader)))
    for sample_idxs, batch_idxs, batch in dataloader:
        assert (batch.shape[1] == n)
        assert (batch.shape[0] <= sample_size)
        assert (batch.shape[2] <= batch_size)
        assert np.alltrue(Y[sample_pool][:, :, batch_pool][sample_idxs]
                          [:, :, batch_idxs] == batch)


if __name__ == "__main__":
    test_sampler()
    test_sampler_pool()