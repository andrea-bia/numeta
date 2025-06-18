import numeta as nm
import numpy as np


def test_comptime_type_hint():

    n = 100
    m = 3
    
    @nm.jit
    def sum_first_n(length: nm.comptime, a, result):
        result[:] = 0.0
        for i in range(length):
            result[:] += a[i]
    
    array = np.random.random((n,))
    result = np.zeros((1,), dtype=array.dtype)

    for i in range(m):
        sum_first_n(i, array, result)
        np.testing.assert_allclose(result, array[:i].sum(), rtol=10e2 * np.finfo(array.dtype).eps)
