import numeta as nm
import numpy as np
import pytest


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_zeros(dtype):
    n = 50
    m = 20

    @nm.jit
    def copy_and_set_zero_first_col_with_zeros(a):
        tmp = nm.zeros((n, m), dtype)

        for i in nm.range(n):
            for j in nm.range(m):
                a[i, j] = tmp[i, j]

    a = np.ones((n, m)).astype(dtype)
    copy_and_set_zero_first_col_with_zeros(a)

    c = np.zeros((n, m)).astype(dtype)

    np.testing.assert_allclose(a, c)


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_zeros_fortran(dtype):
    n = 50
    m = 20

    @nm.jit
    def copy_and_set_zero_first_col_with_zeros(a):
        tmp = nm.zeros((n, m), dtype, order="F")

        for i in nm.range(n):
            for j in nm.range(m):
                a[i, j] = tmp[i, j]

    a = np.ones((n, m)).astype(dtype)
    copy_and_set_zero_first_col_with_zeros(a)

    c = np.zeros((n, m)).astype(dtype)

    np.testing.assert_allclose(a, c)
