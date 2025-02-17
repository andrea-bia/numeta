import numeta as nm
import numpy as np
import pytest


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_pmul(dtype):
    n = 100

    @nm.jit
    def pmul(a, b, c):
        for i in nm.prange(
            a.shape[0],
            default="private",
            shared=[a, b, c, b.shape[0].variable, a.shape[0].variable],
            schedule="static",
        ):
            for k in nm.frange(b.shape[0]):
                c[i, :] += a[i, k] * b[k, :]

    a = np.random.rand(n, n).astype(dtype)
    b = np.random.rand(n, n).astype(dtype)
    c = np.zeros((n, n), dtype=dtype)

    pmul(a, b, c)

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_allclose(c, np.dot(a, b), atol=0)
    else:
        np.testing.assert_allclose(c, np.dot(a, b), rtol=10e2 * np.finfo(dtype).eps)
