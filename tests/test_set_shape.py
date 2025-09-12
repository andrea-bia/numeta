import numpy as np
import numeta as nm


def test_set_shape_literal():

    nm.settings.unset_add_shape_descriptors()

    @nm.jit
    def callee(n, a):
        # if shape descriptors were not disable i should have done a[0, 0]
        a._set_shape((5, 5))
        a[0, 0] = n

    a = np.zeros((5, 5), dtype=np.int64)
    callee(3, a)

    expected = np.zeros((5, 5), dtype=np.int64)
    expected[0, 0] = 3
    np.testing.assert_equal(a, expected)

    nm.settings.set_add_shape_descriptors()


def test_set_shape_variable():

    nm.settings.unset_add_shape_descriptors()

    @nm.jit
    def callee(n, a):
        # if shape descriptors were not disable i should have done a[0, 0]
        a._set_shape((n, n))
        a[1, 2] = 3

    n = 5
    a = np.zeros((n, n), dtype=np.int64)
    callee(n, a)

    expected = np.zeros((n, n), dtype=np.int64)
    expected[1, 2] = 3
    np.testing.assert_equal(a, expected)

    nm.settings.set_add_shape_descriptors()
