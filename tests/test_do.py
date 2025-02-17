import numeta as nm
import numpy as np
import pytest


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_do(dtype):
    n = 100

    @nm.jit
    def do_loop(n, a) -> None:
        i = nm.scalar(nm.i8)
        with nm.do(i, 0, n - 1):
            a[i] = i * 2

    a = np.empty(n, dtype=dtype)

    do_loop(n, a)

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_allclose(a, np.array([i * 2 for i in range(n)]).astype(dtype), atol=0)
    else:
        np.testing.assert_allclose(
            a,
            np.array([i * 2 for i in range(n)]).astype(dtype),
            rtol=10e2 * np.finfo(dtype).eps,
        )
