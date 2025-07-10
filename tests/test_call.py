import numpy as np
import numeta as nm


def test_call_array_scalar():

    @nm.jit
    def callee(n, a):
        a[:] = n

    @nm.jit
    def caller(n, a):
        callee(n, a)

    a = np.zeros((), dtype=np.int64)
    caller(1, a)

    expected = np.zeros((), dtype=np.int64)
    expected[...] = 1
    np.testing.assert_equal(a, expected)


def test_call_array():

    @nm.jit
    def callee(n, a):
        a[:, 2] = n

    @nm.jit
    def caller(n, a):
        callee(n, a)

    a = np.zeros((5, 10), dtype=np.int64)
    caller(1, a)

    expected = np.zeros((5, 10), dtype=np.int64)
    expected[:, 2] = 1
    np.testing.assert_equal(a, expected)


def test_call_getitem_scalar():

    @nm.jit
    def callee(n, a):
        a[:] = n

    @nm.jit
    def caller(n, a):
        callee(n, a[3, 7])

    a = np.zeros((5, 10), dtype=np.int64)
    caller(1, a)

    expected = np.zeros((5, 10), dtype=np.int64)
    expected[3, 7] = 1
    np.testing.assert_equal(a, expected)


def test_call_getitem_slice():

    @nm.jit(directory="test_call_getitem_slice")
    def callee(n, a):
        a[:] = n

    @nm.jit
    def caller(n, a):
        callee(50, a[:])
        callee(n, a[:2, :3])
        callee(n + 1, a[3:4, 8:])
        callee(n + 2, a[:, 7])

    a = np.zeros((5, 10), dtype=np.int64)
    caller(1, a)

    expected = np.zeros((5, 10), dtype=np.int64)
    expected[:] = 50
    expected[:2, :3] = 1
    expected[3:4, 8:] = 2
    expected[:, 7] = 3
    np.testing.assert_equal(a, expected)


def test_call_getitem_slice_runtime_dep():

    @nm.jit(directory="test_call_getitem_slice")
    def callee(n, a):
        a[:] = n

    @nm.jit
    def caller(n, m, a):
        callee(50, a[:])
        callee(n, a[:n, : m // 2])
        callee(2, a[2 : n + 1, m - 2 :])
        callee(3, a[:, n])
        callee(1, a[:, n : m - n])

    n = 4
    m = 9
    a = np.zeros((5, 10), dtype=np.int64)
    caller(n, m, a)

    expected = np.zeros((5, 10), dtype=np.int64)
    expected[:] = 50
    expected[:n, : m // 2] = n
    expected[2 : n + 1, m - 2 :] = 2
    expected[:, n] = 3
    expected[:, n : m - n] = 1
    np.testing.assert_equal(a, expected)


def test_call_getattr_scalar():

    @nm.jit
    def callee(n, a):
        a[:] = n

    @nm.jit
    def caller(n, a):
        callee(n, a["x"])

    dtype = np.dtype([("x", np.int64), ("y", np.float64, (2, 2))], align=True)

    a = np.zeros((), dtype=dtype)
    caller(1, a)

    expected = np.zeros((), dtype=dtype)
    expected["x"] = 1
    np.testing.assert_equal(a, expected)


def test_call_getattr_array():

    @nm.jit
    def callee(n, a):
        a[:] = n

    @nm.jit
    def caller(n, a):
        callee(n, a["y"])

    dtype = np.dtype([("x", np.int64), ("y", np.float64, (2, 2))], align=True)

    a = np.zeros((), dtype=dtype)
    caller(2.0, a)

    expected = np.zeros((), dtype=dtype)
    expected["y"] = 2.0
    np.testing.assert_equal(a, expected)
