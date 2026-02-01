import numpy as np
import numeta as nm


def test_disable_array_descriptor(backend, backend):

    nm.settings.unset_add_shape_descriptors()

    @nm.jit(backend=backend)
    def callee(n, a):
        # if shape descriptors were not disable i should have done a[0, 0]
        a[0] = n

    a = np.zeros((5, 5), dtype=np.int64)
    callee(3, a)

    expected = np.zeros((5, 5), dtype=np.int64)
    expected[0, 0] = 3
    np.testing.assert_equal(a, expected)

    nm.settings.set_add_shape_descriptors()
