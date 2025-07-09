import numeta as nm
import numpy as np
import pytest


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_scalar_array(dtype):

    @nm.jit
    def fill(a):
        a[:] = 2

    a = np.empty(()).astype(dtype)
    fill(a)

    np.testing.assert_allclose(a, np.array([2]).astype(dtype))


def test_scalar():

    @nm.jit
    def fill(a, b):
        a[:] = b

    a = np.empty((), dtype=np.int32)
    b = np.int32(7)
    fill(a, b)

    np.testing.assert_allclose(a, np.array([7], dtype=np.int32))
