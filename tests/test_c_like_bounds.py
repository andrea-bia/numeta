import numpy as np
import numeta as nm
from numeta.syntax.settings import settings


def test_slice_exclusive_bounds():

    @nm.jit
    def fill(a):
        a[1:5] = 1

    arr = np.zeros(6, dtype=np.int64)
    fill(arr)

    expected = np.zeros(6, dtype=np.int64)
    expected[1:5] = 1
    np.testing.assert_array_equal(arr, expected)


def test_slice_inclusive_bounds():

    settings.unset_c_like_bounds()

    @nm.jit
    def fill(a):
        a[:4] = 1

    arr = np.zeros(6, dtype=np.int64)
    fill(arr)

    settings.set_c_like_bounds()

    expected = np.zeros(6, dtype=np.int64)
    expected[:5] = 1
    np.testing.assert_array_equal(arr, expected)


test_slice_inclusive_bounds()
