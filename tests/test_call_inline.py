import numpy as np
import numeta as nm


def test_inline_array_scalar():

    @nm.jit(inline=True)
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


def test_inline_array():

    @nm.jit(inline=True)
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


def test_inline_getitem_scalar():

    @nm.jit(inline=True)
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


def test_inline_getitem_slice():

    @nm.jit(inline=True)
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


def test_inline_getitem_slice_runtime_dep():

    @nm.jit(inline=True)
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


def test_inline_getattr_scalar():

    @nm.jit(inline=True)
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


def test_inline_getattr_array():

    @nm.jit(inline=True)
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


def test_inline_matmul():

    @nm.jit(inline=True)
    def callee(a, d):
        a[:] = d

    @nm.jit
    def caller(a, b, c):
        callee(c, nm.matmul(b, a))

    n = 10
    a = np.random.random((n, n))
    b = np.random.random((n, n))
    c = np.zeros((10, 10))
    caller(a, b, c)

    expected = a @ b
    np.testing.assert_equal(c, expected)


def test_inline_nested_calls():
    @nm.jit(inline=True)
    def inner(n, arr):
        arr[0] = n

    @nm.jit(inline=True)
    def middle(n, arr):
        inner(n, arr)
        arr[1] = n + 1

    @nm.jit
    def caller(n, arr):
        middle(n, arr)

    arr = np.zeros(2, dtype=np.int64)
    caller(5, arr)

    expected = np.array([5, 6], dtype=np.int64)
    np.testing.assert_equal(arr, expected)


def test_inline_nested_calls_dependencies():
    """Test to check that inline nested calls propagate dependencies correctly."""

    @nm.jit
    def inner(n, arr):
        arr[0] = n

    @nm.jit(inline=True)
    def middle(n, arr):
        inner(n, arr)
        arr[1] = n + 1

    @nm.jit
    def caller(n, arr):
        middle(n, arr)

    arr = np.zeros(2, dtype=np.int64)
    caller(5, arr)

    expected = np.array([5, 6], dtype=np.int64)
    np.testing.assert_equal(arr, expected)


def test_inline_matmul_fortran_order():

    @nm.jit(inline=True)
    def callee(a, d):
        a[:] = d

    @nm.jit
    def caller(a, b, c):
        callee(c, nm.matmul(a, b))

    n = 10
    a = np.random.random((n, n)).astype(np.float64, order="F")
    b = np.random.random((n, n)).astype(np.float64, order="F")
    c = np.zeros((10, 10), order="F")
    caller(a, b, c)

    expected = a @ b
    np.testing.assert_equal(c, expected)


def test_inline_loop_matmul():
    @nm.jit(inline=True)
    def callee(n, a, b, c):
        for i in nm.range(n):
            for k in nm.range(n):
                c[i, :] += a[i, k] * b[k, :]

    @nm.jit
    def caller(n, a, b, c):
        callee(n, a, b, c)

    n = 4
    a = np.random.random((n, n))
    b = np.random.random((n, n))
    c = np.zeros((n, n))
    caller(n, a, b, c)

    expected = a @ b
    np.testing.assert_allclose(c, expected)


def test_inline_loop_if():
    @nm.jit(inline=True)
    def callee(n, arr):
        for i in nm.range(n):
            with nm.If(i < 2):
                arr[i] = i
            with nm.ElseIf(i < n // 2):
                arr[i] = i + 1
            with nm.Else():
                arr[i] = -i

    @nm.jit
    def caller(n, arr):
        callee(n, arr)

    n = 10
    arr = np.zeros(n, dtype=np.int64)
    caller(n, arr)

    expected = np.array([0, 1, 3, 4, 5, -5, -6, -7, -8, -9], dtype=np.int64)
    np.testing.assert_equal(arr, expected)


def test_inline_jit_threshold_inline():
    @nm.jit(inline=2)
    def callee(a):
        a[:] = 1
        a[:] = 2

    @nm.jit
    def caller(a):
        callee(a)

    arr = np.zeros(5, dtype=np.int64)
    caller(arr)

    expected = np.full(5, 2, dtype=np.int64)
    np.testing.assert_equal(arr, expected)
    assert caller._NumetaFunction__dependencies == {}


def test_inline_jit_threshold_call():
    @nm.jit(inline=1)
    def callee(a):
        a[:] = 1
        a[:] = 2

    @nm.jit
    def caller(a):
        callee(a)

    arr = np.zeros(5, dtype=np.int64)
    caller(arr)

    expected = np.full(5, 2, dtype=np.int64)
    np.testing.assert_equal(arr, expected)
    assert caller._NumetaFunction__dependencies != {}


def test_inline_jit_threshold_loop_inline():
    @nm.jit(inline=3)
    def callee(n, a):
        for i in nm.range(n):
            for j in nm.range(n):
                a[i, j] += 1

    @nm.jit
    def caller(n, a):
        callee(n, a)

    arr = np.zeros((3, 3), dtype=np.int64)
    caller(3, arr)

    expected = np.ones((3, 3), dtype=np.int64)
    np.testing.assert_equal(arr, expected)
    assert caller._NumetaFunction__dependencies == {}


def test_inline_jit_threshold_loop_call():
    @nm.jit(inline=2)
    def callee(n, a):
        for i in nm.range(n):
            for j in nm.range(n):
                a[i, j] += 1

    @nm.jit
    def caller(n, a):
        callee(n, a)

    arr = np.zeros((3, 3), dtype=np.int64)
    caller(3, arr)

    expected = np.ones((3, 3), dtype=np.int64)
    np.testing.assert_equal(arr, expected)
    assert caller._NumetaFunction__dependencies != {}


def test_inline_jit_threshold_if_inline():
    @nm.jit(inline=5)
    def callee(n, a):
        for i in nm.range(n):
            with nm.If(i < 2):
                a[i] = i
            with nm.Else():
                a[i] = -i

    @nm.jit
    def caller(n, a):
        callee(n, a)

    arr = np.zeros(5, dtype=np.int64)
    caller(5, arr)

    expected = np.array([0, 1, -2, -3, -4], dtype=np.int64)
    np.testing.assert_equal(arr, expected)
    assert caller._NumetaFunction__dependencies == {}


def test_inline_jit_threshold_if_call():
    @nm.jit(inline=4)
    def callee(n, a):
        for i in nm.range(n):
            with nm.If(i < 2):
                a[i] = i
            with nm.Else():
                a[i] = -i

    @nm.jit
    def caller(n, a):
        callee(n, a)

    arr = np.zeros(5, dtype=np.int64)
    caller(5, arr)

    expected = np.array([0, 1, -2, -3, -4], dtype=np.int64)
    np.testing.assert_equal(arr, expected)
    assert caller._NumetaFunction__dependencies != {}
