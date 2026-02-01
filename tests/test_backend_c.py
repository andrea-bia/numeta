import numeta as nm
import numpy as np


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
