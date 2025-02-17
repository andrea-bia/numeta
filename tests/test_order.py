import numeta as nm
import numpy as np
import pytest


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_mul(dtype):
    n = 100

    @nm.jit
    def mul(a, b, c):
        for i in nm.frange(a.shape[0]):
            for k in nm.frange(b.shape[0]):
                c[i, :] += a[i, k] * b[k, :]

    a = np.asfortranarray(np.random.rand(n, n).astype(dtype))
    b = np.random.rand(n, n).astype(dtype)
    c = np.asfortranarray(np.zeros((n, n), dtype=dtype))

    mul(a, b, c)

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_allclose(c, np.dot(a, b), atol=0)
    else:
        np.testing.assert_allclose(c, np.dot(a, b), rtol=10e2 * np.finfo(dtype).eps)
