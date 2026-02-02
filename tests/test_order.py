import numeta as nm
import numpy as np
import pytest


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_mul(dtype, backend):
    n = 100

    @nm.jit(backend=backend)
    def mul(a, b, c):
        for i in nm.range(a.shape[0]):
            for k in nm.range(b.shape[0]):
                c[i, :] += a[i, k] * b[k, :]

    a = np.asfortranarray(np.random.rand(n, n).astype(dtype))
    b = np.random.rand(n, n).astype(dtype)
    c = np.asfortranarray(np.zeros((n, n), dtype=dtype))

    mul(a, b, c)

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_allclose(c, np.dot(a, b), atol=0)
    else:
        np.testing.assert_allclose(c, np.dot(a, b), rtol=10e2 * np.finfo(dtype).eps)


def test_intrinsics_shape_size_rank(backend):
    @nm.jit(backend=backend)
    def info(a, out_shape):
        out_shape[:] = nm.shape(a)
        return nm.size(a, 1) * nm.size(a, 2) + nm.size(a, 1) + nm.rank(a)

    a = np.zeros((2, 2), dtype=np.float64)
    out_shape = np.zeros(2, dtype=np.int64)
    result = info(a, out_shape)
    np.testing.assert_array_equal(out_shape, np.array([2, 2], dtype=np.int64))
    np.testing.assert_equal(result, a.size + a.shape[0] + 2)
