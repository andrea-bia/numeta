import numeta as nm
import numpy as np
import pytest


NUMERIC_DTYPES = [
    np.float64,
    np.float32,
    np.int64,
    np.int32,
    np.complex64,
    np.complex128,
    nm.float64,
    nm.float32,
    nm.int64,
    nm.int32,
    nm.complex64,
    nm.complex128,
]
if hasattr(np, "float128"):
    NUMERIC_DTYPES.extend([np.float128, nm.float128])


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_zeros(dtype, backend):
    n = 50
    m = 20

    @nm.jit(backend=backend)
    def copy_and_set_zero_first_col_with_zeros(a):
        tmp = nm.zeros((n, m), dtype)

        for i in nm.range(n):
            for j in nm.range(m):
                a[i, j] = tmp[i, j]

    np_dtype = nm.get_datatype(dtype).get_numpy()
    a = np.ones((n, m)).astype(np_dtype)
    copy_and_set_zero_first_col_with_zeros(a)

    c = np.zeros((n, m)).astype(np_dtype)

    np.testing.assert_allclose(a, c)


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_zeros_fortran(dtype, backend):
    n = 50
    m = 20

    @nm.jit(backend=backend)
    def copy_and_set_zero_first_col_with_zeros(a):
        tmp = nm.zeros((n, m), dtype, order="F")

        for i in nm.range(n):
            for j in nm.range(m):
                a[i, j] = tmp[i, j]

    np_dtype = nm.get_datatype(dtype).get_numpy()
    a = np.ones((n, m)).astype(np_dtype)
    copy_and_set_zero_first_col_with_zeros(a)

    c = np.zeros((n, m)).astype(np_dtype)

    np.testing.assert_allclose(a, c)
