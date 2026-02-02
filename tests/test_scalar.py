import numeta as nm
import numpy as np
import pytest
from numeta.syntax.expressions import ArrayConstructor


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_scalar(dtype, backend):
    @nm.jit(backend=backend)
    def fill(a):
        tmp = nm.scalar(dtype, 50)
        a[0] = tmp

        tmp2 = nm.scalar(dtype, 100)
        a[1] = tmp2

    a = np.empty(2).astype(dtype)
    fill(a)

    np.testing.assert_allclose(a, [50, 100])


def test_scalar_arithmetic(backend):
    @nm.jit(backend=backend)
    def add_mul(a, b):
        c = a + b
        return c * 2

    result = add_mul(3, 4)
    np.testing.assert_equal(result, 14)


def test_intrinsics_scalar_math(backend):
    @nm.jit(backend=backend)
    def compute(a):
        return nm.abs(a) + nm.sqrt(a)

    result = compute(9.0)
    np.testing.assert_equal(result, 12)


def test_complex_ops(backend):
    @nm.jit(backend=backend)
    def compute(a, b):
        return a + b * (1 + 2j)

    result = compute(1 + 1j, 2 - 1j)
    expected = (1 + 1j) + (2 - 1j) * (1 + 2j)
    np.testing.assert_allclose(result, expected)


def test_complex_parts(backend):
    @nm.jit(backend=backend)
    def combine(a):
        return nm.real(a) + nm.imag(a)

    result = combine(3 + 4j)
    np.testing.assert_equal(result, 7)


def test_array_constructor_assignment(backend):
    @nm.jit(backend=backend)
    def fill(a):
        a[:] = ArrayConstructor(1, 2, 3, 4)

    a = np.zeros(4, dtype=np.int64)
    fill(a)
    np.testing.assert_array_equal(a, np.array([1, 2, 3, 4], dtype=np.int64))


def test_broadcast_scalar(backend):
    @nm.jit(backend=backend)
    def add_scalar(a, b):
        a[:] = a + b

    a = np.arange(4, dtype=np.float64)
    add_scalar(a, 2.5)
    np.testing.assert_allclose(a, np.arange(4, dtype=np.float64) + 2.5)


def test_intrinsics_reductions(backend):
    @nm.jit(backend=backend)
    def compute(a):
        return nm.sum(a) + nm.maxval(a) - nm.minval(a)

    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    result = compute(arr)
    np.testing.assert_allclose(result, arr.sum() + arr.max() - arr.min())


def test_intrinsics_all(backend):
    @nm.jit(backend=backend)
    def check(a):
        return nm.all(a)

    arr = np.array([True, True, False], dtype=bool)
    np.testing.assert_equal(check(arr), False)


def test_intrinsics_bitwise_and_math(backend):
    @nm.jit(backend=backend)
    def bits(a, b, s):
        return nm.iand(a, b) + nm.ior(a, b) + nm.xor(a, b) + nm.ishft(a, s)

    @nm.jit(backend=backend)
    def bits2(a, p):
        return nm.ibset(a, p) + nm.ibclr(a, p) + nm.popcnt(a) + nm.trailz(a)

    @nm.jit(backend=backend)
    def math_ops(a, b):
        return nm.atan2(a, b) + nm.floor(a) + nm.sinh(a) + nm.cosh(a) + nm.tanh(a)

    np.testing.assert_equal(bits(3, 5, 1), (3 & 5) + (3 | 5) + (3 ^ 5) + (3 << 1))
    np.testing.assert_equal(bits2(3, 1), (3 | (1 << 1)) + (3 & ~(1 << 1)) + 2 + 0)

    result = math_ops(1.2, 2.3)
    expected = np.arctan2(1.2, 2.3) + np.floor(1.2) + np.sinh(1.2) + np.cosh(1.2) + np.tanh(1.2)
    np.testing.assert_allclose(result, expected)
