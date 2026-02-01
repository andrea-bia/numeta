# import numeta as nm
# import numpy as np
# import pytest
#
#
# @pytest.mark.parametrize(
#    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
# )
# def test_mul(dtype, backend):
#    n = 100
#
#    @nm.jit(backend=backend)
#    def mul(a: nm.dtype[dtype][:, :], b: nm.dtype[dtype][:, :], c: nm.dtype[dtype][:, :], e, prefactor=0.0, f:nm.dtype[dtype]=8.0):
#
#        if prefactor != 0.0:
#            for i in nm.frange(a.shape[0]):
#                for k in nm.frange(b.shape[0]):
#                    c[i, :] += prefactor * a[i, k] * b[k, :]
#
#        for i in nm.frange(a.shape[0]):
#            for k in nm.frange(b.shape[0]):
#                c[i, :] += a[i, k] * b[k, :]
#
#    a = np.random.rand(n, n).astype(dtype)
#    b = np.random.rand(n, n).astype(dtype)
#    c = np.zeros((n, n), dtype=dtype)
#
#    mul(a, b, c)
#
#    if np.issubdtype(dtype, np.integer):
#        np.testing.assert_allclose(c, np.dot(a, b), atol=0)
#    else:
#        np.testing.assert_allclose(c, np.dot(a, b), rtol=10e2 * np.finfo(dtype).eps)
#
# test_mul(np.float64)
