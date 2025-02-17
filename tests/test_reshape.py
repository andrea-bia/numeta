import numeta as nm
import numpy as np
import pytest


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_reshape(dtype):
    n = 100
    m = 20

    @nm.jit
    def set_zero_first_col(n, m, a):
        a_p = nm.reshape(a, (n, m))

        a_p[:, 0] = 0

    a = np.random.rand(n, m).astype(dtype)
    b = a.copy()

    set_zero_first_col(n, m, a)
    b[:, 0] = 0

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_allclose(a, b, atol=0)
    else:
        np.testing.assert_allclose(a, b, rtol=10e2 * np.finfo(dtype).eps)


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_reshape_fortran(dtype):
    n = 100
    m = 20

    @nm.jit
    def set_zero_first_col(n: nm.i8, m: nm.i8, a: nm.dtype[dtype]["F"][:, :]):
        a_p = nm.reshape(a, (n, m), order="F")

        a_p[:, 0] = 0

    a = np.asfortranarray(np.random.rand(n, m).astype(dtype))
    b = a.copy()

    set_zero_first_col(n, m, a)
    b[:, 0] = 0

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_allclose(a, b, atol=0)
    else:
        np.testing.assert_allclose(a, b, rtol=10e2 * np.finfo(dtype).eps)
