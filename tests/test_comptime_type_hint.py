import numeta as nm
import numpy as np
import pytest


def test_comptime_type_hint():
    n = 100

    dtype = np.float64

    @nm.jit
    def mul(dtype: nm.comptime, a, b, c):
        for i in nm.frange(a.shape[0]):
            for k in nm.frange(b.shape[0]):
                c[i, :] += a[i, k] * b[k, :]

    a = np.random.rand(n, n).astype(dtype)
    b = np.random.rand(n, n).astype(dtype)
    c = np.zeros((n, n), dtype=dtype)

    mul(np.float64, a, b, c)

    np.testing.assert_allclose(c, np.dot(a, b), rtol=10e2 * np.finfo(dtype).eps)

    dtype = np.complex128

    a = np.random.rand(n, n).astype(dtype)
    b = np.random.rand(n, n).astype(dtype)
    c = np.zeros((n, n), dtype=dtype)
    mul(np.complex128, a, b, c)

    np.testing.assert_allclose(c, np.dot(a, b), rtol=10e2 * np.finfo(dtype).eps)
