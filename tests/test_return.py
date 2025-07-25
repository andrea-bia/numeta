import numeta as nm
import numpy as np
import pytest


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_matmul_return_1_ndarray(dtype):
    n = 100

    @nm.jit
    def mul(a, b):
        c = nm.zeros((a.shape[0], b.shape[1]), dtype)
        for i in nm.range(a.shape[0]):
            for k in nm.range(b.shape[0]):
                c[i, :] += a[i, k] * b[k, :]
        return c

    a = np.random.rand(n, n).astype(dtype)
    b = np.random.rand(n, n).astype(dtype)

    c = mul(a, b)

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_allclose(c, np.dot(a, b), atol=0)
    else:
        np.testing.assert_allclose(c, np.dot(a, b), rtol=10e2 * np.finfo(dtype).eps)


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_matmul_return_2_ndarray(dtype):
    n = 100

    @nm.jit
    def mul(a, b):
        c = nm.zeros((a.shape[0], b.shape[1]), dtype)
        d = nm.zeros((a.shape[0], b.shape[1]), dtype)
        for i in nm.range(a.shape[0]):
            for k in nm.range(b.shape[0]):
                c[i, :] += a[i, k] * b[k, :]
        return c, d

    a = np.random.rand(n, n).astype(dtype)
    b = np.random.rand(n, n).astype(dtype)

    c, d = mul(a, b)

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_allclose(c, np.dot(a, b), atol=0)
        np.testing.assert_allclose(d, np.zeros((n, n), dtype=dtype), atol=0)
    else:
        np.testing.assert_allclose(c, np.dot(a, b), rtol=10e2 * np.finfo(dtype).eps)
        np.testing.assert_allclose(
            d, np.zeros((n, n), dtype=dtype), rtol=10e2 * np.finfo(dtype).eps
        )


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_return_scalar(dtype):

    @nm.jit
    def return_scalar():
        return nm.scalar(dtype, 42)

    scalar = return_scalar()

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_allclose(scalar, 42, atol=0)
    else:
        np.testing.assert_allclose(scalar, 42, rtol=10e2 * np.finfo(dtype).eps)
