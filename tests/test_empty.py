import numeta as nm
import numpy as np
import pytest

@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_empty(dtype):
    n = 50

    @nm.jit
    def copy_with_empty(a: nm.dtype[dtype][:, :], b: nm.dtype[dtype][:, :]):

        tmp = nm.empty((a.shape[0], a.shape[1]), nm.dtype[dtype])
        tmp[:] = a
        b[:] = tmp

    a = np.ones((n, n)).astype(dtype)
    b = np.zeros((n, n)).astype(dtype)
    copy_with_empty(a, b)

    np.testing.assert_allclose(a, b)
