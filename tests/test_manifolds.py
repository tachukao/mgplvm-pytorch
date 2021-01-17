from mgplvm import manifolds


def test_euclid_dimensions():
    e3 = manifolds.Euclid(10, 3)
    assert e3.d == 3
    assert e3.m == 10


def test_torus_dimensions():
    t3 = manifolds.Torus(10, 3)
    assert t3.d == 3
    assert t3.m == 10


def test_so3_dimensions():
    so3 = manifolds.So3(10)
    assert so3.d == 3
    assert so3.m == 10
    
if __name__ == '__main__':
    test_euclid_dimensions()
    test_torus_dimensions()
    test_so3_dimensions()
    print('Tested manifolds')
    