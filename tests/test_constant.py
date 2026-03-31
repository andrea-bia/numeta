from pathlib import Path

import numpy as np
import pytest

import numeta as nm


HAS_FLOAT128 = hasattr(np, "float128")
HAS_COMPLEX256 = hasattr(np, "complex256")


# ──────────────────────────────────────────────────────────────────────────────
# Float precision matrix
# ──────────────────────────────────────────────────────────────────────────────
FLOAT_CASES = [
    pytest.param(np.float32, np.float32(1.2345679), "_c_float", "npy_float32", id="float32"),
    pytest.param(
        np.float64,
        np.float64(1.2345678901234567),
        "_c_double",
        "npy_float64",
        id="float64",
    ),
    pytest.param(
        getattr(np, "float128", np.float64),
        getattr(np, "float128", np.float64)(1.2345678901234567),
        "_c_long_double",
        "npy_longdouble",
        id="float128",
        marks=pytest.mark.skipif(
            not HAS_FLOAT128, reason="np.float128 is not available on this platform"
        ),
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# Integer precision matrix
# ──────────────────────────────────────────────────────────────────────────────
INT_CASES = [
    pytest.param(np.int32, np.int32(42), "_c_int32_t", "npy_int32", id="int32"),
    pytest.param(np.int64, np.int64(42), "_c_int64_t", "npy_int64", id="int64"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Complex precision matrix
# ──────────────────────────────────────────────────────────────────────────────
COMPLEX_CASES = [
    pytest.param(
        np.complex64,
        np.complex64(1.2345679 + 2.718281j),
        "_c_float_complex",
        "npy_complex64",
        id="complex64",
    ),
    pytest.param(
        np.complex128,
        np.complex128(1.2345678901234567 + 2.718281828459045j),
        "_c_double_complex",
        "npy_complex128",
        id="complex128",
    ),
    pytest.param(
        getattr(np, "complex256", np.complex128),
        getattr(np, "complex256", np.complex128)(1.234567890123456789 + 2.7182818284590452353j),
        "_c_long_double_complex",
        "npy_clongdouble",
        id="complex256",
        marks=pytest.mark.skipif(
            not HAS_COMPLEX256, reason="np.complex256 is not available on this platform"
        ),
    ),
]


def _read_generated_source(tmp_path: Path, func_name: str, backend: str) -> str:
    suffix = "f90" if backend == "fortran" else "c"
    return (tmp_path / f"{func_name}_src.{suffix}").read_text()


# ──────────────────────────────────────────────────────────────────────────────
# Scalar constants
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("dtype, value, fortran_kind, c_type", FLOAT_CASES)
def test_constant_scalar_float(tmp_path, backend, dtype, value, fortran_kind, c_type):
    dtype_name = np.dtype(dtype).name
    func_name = f"const_scalar_{backend}_{dtype_name}"

    @nm.jit(backend=backend, directory=str(tmp_path), namer=lambda *spec: func_name)
    def fill(a):
        c = nm.constant(value, dtype=dtype, name="c_scalar")
        a[0] = c

    arr = np.zeros(1, dtype=dtype)
    fill(arr)

    assert arr[0] == value

    source = _read_generated_source(tmp_path, func_name, backend)
    if backend == "fortran":
        assert "data c_scalar" in source
        assert fortran_kind in source
    else:
        assert f"{c_type} c_scalar1" in source


@pytest.mark.parametrize("dtype, value, fortran_kind, c_type", INT_CASES)
def test_constant_scalar_int(tmp_path, backend, dtype, value, fortran_kind, c_type):
    dtype_name = np.dtype(dtype).name
    func_name = f"const_scalar_{backend}_{dtype_name}"

    @nm.jit(backend=backend, directory=str(tmp_path), namer=lambda *spec: func_name)
    def fill(a):
        c = nm.constant(value, dtype=dtype, name="c_scalar")
        a[0] = c

    arr = np.zeros(1, dtype=dtype)
    fill(arr)

    assert arr[0] == value

    source = _read_generated_source(tmp_path, func_name, backend)
    if backend == "fortran":
        assert "data c_scalar" in source
        assert fortran_kind in source
    else:
        assert f"{c_type} c_scalar1" in source


@pytest.mark.parametrize("dtype, value, fortran_kind, c_type", COMPLEX_CASES)
def test_constant_scalar_complex(tmp_path, backend, dtype, value, fortran_kind, c_type):
    dtype_name = np.dtype(dtype).name
    func_name = f"const_scalar_{backend}_{dtype_name}"

    @nm.jit(backend=backend, directory=str(tmp_path), namer=lambda *spec: func_name)
    def fill(a):
        c = nm.constant(value, dtype=dtype, name="c_scalar")
        a[0] = c

    arr = np.zeros(1, dtype=dtype)
    fill(arr)

    assert arr[0] == value

    source = _read_generated_source(tmp_path, func_name, backend)
    if backend == "fortran":
        assert "data c_scalar" in source
        assert fortran_kind in source
    else:
        assert f"{c_type} c_scalar1" in source


# ──────────────────────────────────────────────────────────────────────────────
# Array constants
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("dtype, value, fortran_kind, c_type", FLOAT_CASES)
def test_constant_array_float(tmp_path, backend, dtype, value, fortran_kind, c_type):
    values = np.array([value, dtype(2.0)], dtype=dtype)
    dtype_name = np.dtype(dtype).name
    func_name = f"const_array_{backend}_{dtype_name}"

    @nm.jit(backend=backend, directory=str(tmp_path), namer=lambda *spec: func_name)
    def fill(a):
        c = nm.constant(values, dtype=dtype, name="c_array")
        for i in nm.range(2):
            a[i] = c[i]

    arr = np.zeros(2, dtype=dtype)
    fill(arr)

    np.testing.assert_array_equal(arr, values)

    source = _read_generated_source(tmp_path, func_name, backend)
    if backend == "fortran":
        assert "data c_array" in source
        assert fortran_kind in source
    else:
        assert f"{c_type} c_array1[2] = {{" in source


@pytest.mark.parametrize("dtype, value, fortran_kind, c_type", INT_CASES)
def test_constant_array_int(tmp_path, backend, dtype, value, fortran_kind, c_type):
    values = np.array([value, dtype(7)], dtype=dtype)
    dtype_name = np.dtype(dtype).name
    func_name = f"const_array_{backend}_{dtype_name}"

    @nm.jit(backend=backend, directory=str(tmp_path), namer=lambda *spec: func_name)
    def fill(a):
        c = nm.constant(values, dtype=dtype, name="c_array")
        for i in nm.range(2):
            a[i] = c[i]

    arr = np.zeros(2, dtype=dtype)
    fill(arr)

    np.testing.assert_array_equal(arr, values)

    source = _read_generated_source(tmp_path, func_name, backend)
    if backend == "fortran":
        assert "data c_array" in source
        assert fortran_kind in source
    else:
        assert f"{c_type} c_array1[2] = {{" in source


@pytest.mark.parametrize("dtype, value, fortran_kind, c_type", COMPLEX_CASES)
def test_constant_array_complex(tmp_path, backend, dtype, value, fortran_kind, c_type):
    values = np.array([value, dtype(2.0 + 3.0j)], dtype=dtype)
    dtype_name = np.dtype(dtype).name
    func_name = f"const_array_{backend}_{dtype_name}"

    @nm.jit(backend=backend, directory=str(tmp_path), namer=lambda *spec: func_name)
    def fill(a):
        c = nm.constant(values, dtype=dtype, name="c_array")
        for i in nm.range(2):
            a[i] = c[i]

    arr = np.zeros(2, dtype=dtype)
    fill(arr)

    np.testing.assert_array_equal(arr, values)

    source = _read_generated_source(tmp_path, func_name, backend)
    if backend == "fortran":
        assert "data c_array" in source
        assert fortran_kind in source
    else:
        assert f"{c_type} c_array1[2] = {{" in source
