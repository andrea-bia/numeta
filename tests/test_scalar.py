import numeta as nm
import numpy as np
import pytest
from numeta.ast.expressions import ArrayConstructor


def _as_py_complex(value):
    return complex(value)


def _as_np_complex64(value):
    return np.complex64(value)


def _as_np_complex128(value):
    return np.complex128(value)


COMPLEX_SCALAR_CASTERS = [
    pytest.param(_as_py_complex, id="py_complex"),
    pytest.param(_as_np_complex64, id="np_complex64"),
    pytest.param(_as_np_complex128, id="np_complex128"),
]


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


def test_complex_ops(backend):
    @nm.jit(backend=backend)
    def compute(a, b):
        return a + b * (1 + 2j)

    result = compute(1 + 1j, 2 - 1j)
    expected = (1 + 1j) + (2 - 1j) * (1 + 2j)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("cast_a", COMPLEX_SCALAR_CASTERS)
@pytest.mark.parametrize("cast_b", COMPLEX_SCALAR_CASTERS)
def test_complex_ops_full_matrix(cast_a, cast_b, backend):
    @nm.jit(backend=backend)
    def compute(a, b):
        return a + b * (1 + 2j)

    a = cast_a(1 + 1j)
    b = cast_b(2 - 1j)
    result = compute(a, b)
    expected = a + b * (1 + 2j)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("cast", COMPLEX_SCALAR_CASTERS)
def test_complex_parts_scalar_kinds(cast, backend):
    @nm.jit(backend=backend)
    def combine(a):
        return nm.real(a) + nm.imag(a)

    arg = cast(3 + 4j)
    result = combine(arg)
    expected = float(np.real(np.complex128(arg)) + np.imag(np.complex128(arg)))
    np.testing.assert_equal(result, expected)


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
