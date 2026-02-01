import numeta as nm
import numpy as np
from numeta.syntax.expressions import ArrayConstructor


def test_c_backend_scalar_arithmetic():
    @nm.jit(backend="c")
    def add_mul(a, b):
        c = a + b
        return c * 2

    result = add_mul(3, 4)
    np.testing.assert_equal(result, 14)


def test_c_backend_do_loop():
    @nm.jit(backend="c")
    def sum_to(n):
        s = nm.scalar(nm.i8, 0)
        i = nm.scalar(nm.i8)
        with nm.do(i, 0, n - 1):
            s[:] = s + i
        return s

    result = sum_to(5)
    np.testing.assert_equal(result, 10)


def test_c_backend_if_else():
    @nm.jit(backend="c")
    def choose(a, b, flag):
        out = nm.scalar(nm.i8)
        with nm.If(flag):
            out[:] = a
        with nm.Else():
            out[:] = b
        return out

    np.testing.assert_equal(choose(3, 7, True), 3)
    np.testing.assert_equal(choose(3, 7, False), 7)


def test_c_backend_nested_calls():
    @nm.jit(backend="c")
    def inner(a, b):
        return a + b

    @nm.jit(backend="c")
    def outer(a, b, c):
        return inner(a, b) * c

    np.testing.assert_equal(outer(2, 4, 3), 18)


def test_c_backend_intrinsics():
    @nm.jit(backend="c")
    def compute(a):
        return nm.abs(a) + nm.sqrt(a)

    result = compute(9)
    np.testing.assert_equal(result, 12)


def test_c_backend_complex_ops():
    @nm.jit(backend="c")
    def compute(a, b):
        return a + b * (1 + 2j)

    result = compute(1 + 1j, 2 - 1j)
    expected = (1 + 1j) + (2 - 1j) * (1 + 2j)
    np.testing.assert_allclose(result, expected)


def test_c_backend_complex_parts():
    @nm.jit(backend="c")
    def combine(a):
        return a.real + a.imag

    result = combine(3 + 4j)
    np.testing.assert_equal(result, 7)


def test_c_backend_array_constructor_assignment():
    @nm.jit(backend="c")
    def fill(a):
        a[:] = ArrayConstructor(1, 2, 3, 4)

    a = np.zeros(4, dtype=np.int64)
    fill(a)
    np.testing.assert_array_equal(a, np.array([1, 2, 3, 4], dtype=np.int64))


def test_c_backend_broadcast_scalar():
    @nm.jit(backend="c")
    def add_scalar(a, b):
        a[:] = a + b

    a = np.arange(4, dtype=np.float64)
    add_scalar(a, 2.5)
    np.testing.assert_allclose(a, np.arange(4, dtype=np.float64) + 2.5)


def test_c_backend_broadcast_vector():
    @nm.jit(backend="c")
    def add_vec(a, b):
        a[:] = a + b

    a = np.zeros((2, 3), dtype=np.float64, order="C")
    b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    add_vec(a, b)
    np.testing.assert_allclose(a, np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float64))


def test_c_backend_struct_scalar():
    dtype = np.dtype([("x", np.int32), ("y", np.float64)], align=True)

    @nm.jit(backend="c")
    def fill(a):
        a["x"] = 3
        a["y"] = 2.5

    arr = np.zeros(1, dtype=dtype)
    fill(arr[0])

    expected = np.zeros(1, dtype=dtype)
    expected[0]["x"] = 3
    expected[0]["y"] = 2.5
    np.testing.assert_equal(arr, expected)


def test_c_backend_struct_nested_array():
    n = 2
    m = 3

    np_nested1 = np.dtype([("a", np.int64, (n, n)), ("b", np.float64, (m,))], align=True)
    np_nested2 = np.dtype([("c", np_nested1, (n,)), ("d", np_nested1, (3,))], align=True)
    np_nested3 = np.dtype([("c", np_nested2, (2,)), ("d", np_nested1, (3,))], align=True)

    @nm.jit(backend="c")
    def mod_struct(a) -> None:
        a[1]["c"][1]["d"][2]["b"][1] = -4.0

    a = np.zeros(2, dtype=np_nested3)
    mod_struct(a)

    b = np.zeros(2, dtype=np_nested3)
    b[1]["c"][1]["d"][2]["b"][1] = -4.0
    np.testing.assert_equal(a, b)


def test_c_backend_intrinsics_reductions():
    @nm.jit(backend="c")
    def compute(a):
        return nm.sum(a) + nm.maxval(a) - nm.minval(a)

    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    result = compute(arr)
    np.testing.assert_allclose(result, arr.sum() + arr.max() - arr.min())


def test_c_backend_intrinsics_all():
    @nm.jit(backend="c")
    def check(a):
        return nm.all(a)

    arr = np.array([True, True, False], dtype=bool)
    np.testing.assert_equal(check(arr), False)


def test_c_backend_intrinsics_shape_size_rank():
    @nm.jit(backend="c")
    def info(a, out_shape):
        out_shape[:] = nm.shape(a)
        return nm.size(a, 1) * nm.size(a, 2) + nm.size(a, 1) + nm.rank(a)

    a = np.zeros((2, 3), dtype=np.float64)
    out_shape = np.zeros(2, dtype=np.int64)
    result = info(a, out_shape)
    np.testing.assert_array_equal(out_shape, np.array([2, 3], dtype=np.int64))
    np.testing.assert_equal(result, a.size + a.shape[0] + 2)


def test_c_backend_intrinsics_dot_matmul_transpose():
    @nm.jit(backend="c")
    def dot(a, b):
        return nm.dot_product(a, b)

    @nm.jit(backend="c")
    def matmul_ab(a, b, out):
        out[:] = nm.matmul(a, b)

    @nm.jit(backend="c")
    def trans(a, out):
        out[:] = nm.transpose(a)

    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([3.0, 2.0, 1.0], dtype=np.float64)
    np.testing.assert_allclose(dot(a, b), np.dot(a, b))

    m = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    n = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    out = np.zeros((2, 2), dtype=np.float64)
    matmul_ab(m, n, out)
    np.testing.assert_allclose(out, m @ n)

    out_t = np.zeros((2, 2), dtype=np.float64)
    trans(m, out_t)
    np.testing.assert_allclose(out_t, m.T)


def test_c_backend_intrinsics_bitwise_and_math():
    @nm.jit(backend="c")
    def bits(a, b, s):
        return nm.iand(a, b) + nm.ior(a, b) + nm.xor(a, b) + nm.ishft(a, s)

    @nm.jit(backend="c")
    def bits2(a, p):
        return nm.ibset(a, p) + nm.ibclr(a, p) + nm.popcnt(a) + nm.trailz(a)

    @nm.jit(backend="c")
    def math_ops(a, b):
        return nm.atan2(a, b) + nm.floor(a) + nm.sinh(a) + nm.cosh(a) + nm.tanh(a)

    np.testing.assert_equal(bits(3, 5, 1), (3 & 5) + (3 | 5) + (3 ^ 5) + (3 << 1))
    np.testing.assert_equal(bits2(3, 1), (3 | (1 << 1)) + (3 & ~(1 << 1)) + 2 + 0)

    result = math_ops(1.2, 2.3)
    expected = np.arctan2(1.2, 2.3) + np.floor(1.2) + np.sinh(1.2) + np.cosh(1.2) + np.tanh(1.2)
    np.testing.assert_allclose(result, expected)
