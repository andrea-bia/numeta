import numeta as nm
import numpy as np
import pytest

# Type groups
FLOATS = [np.float32, np.float64]
COMPLEX = [np.complex64, np.complex128]
FLOATS_COMPLEX = FLOATS + COMPLEX
INTS = [np.int32, np.int64]


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


def get_args(dtype, *values):
    return tuple(dtype(v) for v in values)


@pytest.mark.parametrize("dtype", FLOATS_COMPLEX)
@pytest.mark.parametrize(
    "func_name, args_raw",
    [
        ("exp", (2.0,)),
        ("log", (2.0,)),
        ("sqrt", (4.0,)),
        ("sin", (1.0,)),
        ("cos", (1.0,)),
        ("tan", (0.5,)),
        ("sinh", (1.0,)),
        ("cosh", (1.0,)),
        ("tanh", (0.5,)),
        ("arcsin", (0.5,)),
        ("arccos", (0.5,)),
        ("arctan", (0.5,)),
        ("arcsinh", (0.5,)),
        ("arccosh", (2.0,)),  # > 1
        ("arctanh", (0.5,)),  # < 1
        ("abs", (-2.5,)),
    ],
)
def test_intrinsics_math_common(dtype, func_name, args_raw, backend):
    if func_name == "abs" and dtype in COMPLEX:
        # abs(complex) returns float (real), not complex.
        # numeta return type inference needs to handle this.
        # existing test_intrinsics_scalar_math used nm.Abs(9.0) -> float.
        pass

    func = getattr(nm, func_name)
    np_func = getattr(np, func_name)
    args = get_args(dtype, *args_raw)

    @nm.jit(backend=backend)
    def run(*a):
        return func(*a)

    res = run(*args)
    expected = np_func(*args)

    # Relax tolerance for complex64/float32
    rtol = 1e-5 if dtype in [np.float32, np.complex64] else 1e-7
    np.testing.assert_allclose(res, expected, rtol=rtol)


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize(
    "func_name, args_raw",
    [
        ("log10", (2.0,)),
        ("floor", (2.7,)),
        ("ceil", (2.3,)),
        ("hypot", (3.0, 4.0)),
        ("copysign", (1.0, -1.0)),
        ("arctan2", (1.0, 2.0)),
    ],
)
def test_intrinsics_math_real(dtype, func_name, args_raw, backend):
    func = getattr(nm, func_name)
    np_func = getattr(np, func_name)
    args = get_args(dtype, *args_raw)

    @nm.jit(backend=backend)
    def run(*a):
        return func(*a)

    res = run(*args)
    expected = np_func(*args)
    rtol = 1e-5 if dtype == np.float32 else 1e-7
    np.testing.assert_allclose(res, expected, rtol=rtol)


def test_intrinsics_reductions(backend):
    @nm.jit(backend=backend)
    def compute(a):
        return nm.sum(a) + nm.max(a) - nm.min(a)

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
        return nm.bitwise_and(a, b) + nm.bitwise_or(a, b) + nm.bitwise_xor(a, b) + nm.ishft(a, s)

    @nm.jit(backend=backend)
    def bits2(a, p):
        return nm.ibset(a, p) + nm.ibclr(a, p) + nm.popcnt(a) + nm.trailz(a)

    @nm.jit(backend=backend)
    def math_ops(a, b):
        return nm.arctan2(a, b) + nm.floor(a) + nm.sinh(a) + nm.cosh(a) + nm.tanh(a)

    np.testing.assert_equal(bits(3, 5, 1), (3 & 5) + (3 | 5) + (3 ^ 5) + (3 << 1))
    np.testing.assert_equal(bits2(3, 1), (3 | (1 << 1)) + (3 & ~(1 << 1)) + 2 + 0)

    result = math_ops(1.2, 2.3)
    expected = np.arctan2(1.2, 2.3) + np.floor(1.2) + np.sinh(1.2) + np.cosh(1.2) + np.tanh(1.2)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("dtype", INTS)
@pytest.mark.parametrize(
    "func_name, val",
    [
        ("exp", 2),
        ("log", 2),
        ("log10", 100),
        ("sqrt", 16),
        ("sin", 0),
        ("cos", 0),
        ("tan", 0),
        ("sinh", 0),
        ("cosh", 0),
        ("tanh", 0),
        ("abs", -5),
    ],
)
def test_integer_inputs(dtype, func_name, val, backend):
    func = getattr(nm, func_name)
    np_func = getattr(np, func_name)

    arg = dtype(val)

    @nm.jit(backend=backend)
    def run(a):
        return func(a)

    res = run(arg)
    expected = np_func(arg)

    np.testing.assert_allclose(res, expected)


@pytest.mark.parametrize("dtype", COMPLEX)
def test_complex_log10(dtype, backend):
    # log10 is tricky for complex in some backends/implementations, checking basic support
    # NumPy supports it. C supports clog10? C99 does not standardly have clog10, it has clog.
    # But numeta might map log10 to log10(real) or custom?
    # Let's include it for float, check complex carefully.
    arg = dtype(10.0 + 10.0j)

    @nm.jit(backend=backend)
    def run(a):
        return nm.log10(a)

    res = run(arg)
    expected = np.log10(arg)
    np.testing.assert_allclose(res, expected, rtol=1e-5)


@pytest.mark.parametrize("cast", COMPLEX_SCALAR_CASTERS)
def test_complex_log10_scalar_kinds(cast, backend):
    arg = cast(10.0 + 10.0j)

    @nm.jit(backend=backend)
    def run(a):
        return nm.log10(a)

    res = run(arg)
    expected = np.log10(arg)
    rtol = 1e-5 if np.asarray(arg).dtype == np.dtype(np.complex64) else 1e-7
    np.testing.assert_allclose(res, expected, rtol=rtol)
