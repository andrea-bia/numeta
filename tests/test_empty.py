import numeta as nm
import numpy as np
import pytest


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_empty(dtype):
    n = 50
    m = 20

    @nm.jit
    def copy_and_set_zero_first_col_with_empty(a: nm.dtype[dtype][:, :], b: nm.dtype[dtype][:, :]):
        tmp = nm.empty((n, m), nm.dtype[dtype])
        tmp[:] = 1.0
        tmp[:, 0] = 0

        for i in nm.frange(n):
            for j in nm.frange(m):
                b[i, j] = tmp[i, j]

    a = np.ones((n, m)).astype(dtype)
    b = np.zeros((n, m)).astype(dtype)
    copy_and_set_zero_first_col_with_empty(a, b)

    c = a.copy()
    c[:, 0] = 0

    np.testing.assert_allclose(b, c)
