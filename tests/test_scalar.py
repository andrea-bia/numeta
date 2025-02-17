import numeta as nm
import numpy as np
import pytest


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_scalar(dtype):
    @nm.jit
    def fill(a):
        tmp = nm.scalar(nm.dtype[dtype], 50)
        a[0] = tmp

        tmp2 = nm.scalar(nm.dtype[dtype], 100)
        a[1] = tmp2

    a = np.empty(2).astype(dtype)
    fill(a)

    np.testing.assert_allclose(a, [50, 100])
