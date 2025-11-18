import numeta as nm
import numpy as np
import pytest


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_cast(dtype):

    @nm.jit
    def set_nine(a):
        a_int = nm.cast(a, dtype)
        a_int[:] = 9.0

    # should contain everything
    a = np.ones(16, dtype=np.bool_)
    set_nine(a)

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_allclose(a.view(dtype)[0], np.array(9, dtype=dtype), atol=0)
    else:
        np.testing.assert_allclose(
            a.view(dtype)[0], np.array(9, dtype=dtype), rtol=10e2 * np.finfo(dtype).eps
        )
