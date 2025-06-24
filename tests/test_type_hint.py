import numeta as nm
import numpy as np
import pytest


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_CompTime(dtype):
    n = 100

    @nm.jit
    def mul(ct: nm.comptime, a, b, c):
        for i in nm.range(a.shape[0]):
            for k in nm.range(b.shape[0]):
                c[i, :] += a[i, k] * b[k, :]

    a = np.random.rand(n, n).astype(dtype)
    b = np.random.rand(n, n).astype(dtype)
    c = np.zeros((n, n), dtype=dtype)

    mul(1.0, a, b, c)

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_allclose(c, np.dot(a, b), atol=0)
    else:
        np.testing.assert_allclose(c, np.dot(a, b), rtol=float(10e2 * np.finfo(dtype).eps))


def test_struct_array():
    n = 2
    m = 3

    np_nested1 = np.dtype([("a", np.int64, (n, n)), ("b", np.float64, (m,))], align=True)
    np_nested2 = np.dtype([("c", np_nested1, (n,)), ("d", np_nested1, (3,))], align=True)
    np_nested3 = np.dtype([("c", np_nested2, (2,)), ("d", np_nested1, (3,))], align=True)

    @nm.jit
    def mod_struct(a) -> None:
        a[1]["c"][1]["d"][2]["b"][1] = -4.0

    a = np.zeros(2, dtype=np_nested3)

    mod_struct(a)

    b = np.zeros(2, dtype=np_nested3)
    b[1]["c"][1]["d"][2]["b"][1] = -4.0

    np.testing.assert_equal(a, b)


def test_struct():
    n = 2
    m = 3

    np_nested1 = np.dtype([("a", np.int64, (n, n)), ("b", np.float64, (m,))], align=True)
    np_nested2 = np.dtype([("c", np_nested1, (n,)), ("d", np_nested1, (3,))], align=True)
    np_nested3 = np.dtype([("c", np_nested2, (2,)), ("d", np_nested1, (3,))], align=True)

    @nm.jit
    def mod_struct(a) -> None:
        a["c"][1]["d"][2]["b"][1] = -4.0

    a = np.zeros(2, dtype=np_nested3)

    mod_struct(a[1])

    b = np.zeros(2, dtype=np_nested3)
    b[1]["c"][1]["d"][2]["b"][1] = -4.0

    np.testing.assert_equal(a, b)
