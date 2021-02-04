import mgplvm as mgp


def test_data_shapes():
    n = 100
    m = 20
    n_samples = 10
    d = 5
    for d, manif in [
        (d, mgp.syndata.Euclid(d)),
        (d, mgp.syndata.Torus(d)),
        (d + 1, mgp.syndata.Sphere(d)),
        (4, mgp.syndata.So3()),
    ]:
        gen = mgp.syndata.Gen(manif, n, m, n_samples=n_samples)
        Y = gen.gen_data()
        assert (Y.shape == (n_samples, n, m))
        assert (gen.gs[0].shape == (n_samples, m, d))


if __name__ == "__main__":
    test_data_shapes()
