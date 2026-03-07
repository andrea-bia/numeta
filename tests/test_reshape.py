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
if hasattr(np, "complex256"):
    NUMERIC_DTYPES.extend([np.complex256, nm.complex256])


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_reshape(dtype, backend):
    n = 100
    m = 20

    @nm.jit(backend=backend)
    def set_zero_first_col(n, m, a):
        a_p = nm.reshape(a, (n, m))

        a_p[:, 0] = 0

    np_dtype = nm.get_datatype(dtype).get_numpy()
    a = np.random.rand(n, m).astype(np_dtype)
    b = a.copy()

    set_zero_first_col(n, m, a)
    b[:, 0] = 0

    if np.issubdtype(np_dtype, np.integer):
        np.testing.assert_allclose(a, b, atol=0)
    else:
        np.testing.assert_allclose(a, b, rtol=10e2 * np.finfo(np_dtype).eps)


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_reshape_fortran(dtype, backend):
    n = 100
    m = 20

    @nm.jit(backend=backend)
    def set_zero_first_col(n, m, a):
        a_p = nm.reshape(a, (n, m), order="F")

        a_p[:, 0] = 0

    np_dtype = nm.get_datatype(dtype).get_numpy()
    a = np.asfortranarray(np.random.rand(n, m).astype(np_dtype))
    b = a.copy()

    set_zero_first_col(n, m, a)
    b[:, 0] = 0

    if np.issubdtype(np_dtype, np.integer):
        np.testing.assert_allclose(a, b, atol=0)
    else:
        np.testing.assert_allclose(a, b, rtol=10e2 * np.finfo(np_dtype).eps)


def test_reshape_accepts_shape_array(backend):
    n = 12
    m = 9

    @nm.jit(backend=backend)
    def reshape_with_shape_array(a, b):
        a_r = nm.reshape(a, nm.Shape(b))
        a_r[:] = 1.0

    a = np.zeros((n, m), dtype=np.float64)
    b = np.zeros((n, m), dtype=np.float64)
    reshape_with_shape_array(a, b)
    np.testing.assert_allclose(a, np.ones((n, m), dtype=np.float64))
