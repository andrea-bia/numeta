# import numeta as nm
# import numeta.experimental as nme
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
#    def mul(a: nm.dtype[dtype][:, :], b: nm.dtype[dtype][:, :], c: nm.dtype[dtype][:, :]):
#        for i in nm.frange(a.shape[0]):
#            for k in nm.frange(b.shape[0]):
#                c[i, :] += a[i, k] * b[k, :]
#
#        d = nme.get_view(c, (c.shape[0], c.shape[1], 100))
#        d[:] = 0.0
#        e = nme.get_view(d, (c.shape[0], c.shape[1]))
#        e[:] = 0.0
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
#
# print(test_mul(np.float64))
