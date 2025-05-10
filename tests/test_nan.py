import numeta as nm
import numpy as np
import pytest


def test_nan():

    @nm.jit
    def check_nan(a, is_nan):
        is_nan[0] = a[0] != a[0]

    a = np.array((1,), dtype=np.float64)
    a[0] = np.nan

    is_nan = np.zeros((1,), dtype=np.bool_)

    check_nan(a, is_nan)

    np.testing.assert_equal(is_nan[0], a[0] != a[0])
