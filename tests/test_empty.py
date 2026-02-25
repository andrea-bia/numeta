import numeta as nm
import numpy as np
import pytest
from numeta.array_shape import ArrayShape
from numeta.ast.variable import Variable
from numeta.builder_helper import BuilderHelper


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_empty(dtype, backend):
    n = 50
    m = 20

    @nm.jit(backend=backend)
    def copy_and_set_zero_first_col_with_empty(a, b):
        tmp = nm.empty((n, m), dtype)
        tmp[:] = 1.0
        tmp[:, 0] = 0

        for i in nm.range(n):
            for j in nm.range(m):
                b[i, j] = tmp[i, j]

    a = np.ones((n, m)).astype(dtype)
    b = np.zeros((n, m)).astype(dtype)
    copy_and_set_zero_first_col_with_empty(a, b)

    c = a.copy()
    c[:, 0] = 0

    np.testing.assert_allclose(b, c)


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_empty_runtime_shape(dtype, backend):
    n = 50
    m = 20

    @nm.jit(backend=backend)
    def copy_and_set_zero_first_col_with_empty(a, b, n, m):
        tmp = nm.empty((n, m), dtype)
        tmp[:] = 1.0
        tmp[:, 0] = 0

        for i in nm.range(n):
            for j in nm.range(m):
                b[i, j] = tmp[i, j]

    a = np.ones((n, m)).astype(dtype)
    b = np.zeros((n, m)).astype(dtype)
    copy_and_set_zero_first_col_with_empty(a, b, n, m)

    c = a.copy()
    c[:, 0] = 0

    np.testing.assert_allclose(b, c)


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_empty_fortran(dtype, backend):
    n = 50
    m = 20

    @nm.jit(backend=backend)
    def copy_and_set_zero_first_col_with_empty(a, b):
        tmp = nm.empty((n, m), dtype, order="F")
        tmp[:] = 1.0
        tmp[:, 0] = 0

        tmp_p = nm.reshape(tmp, n * m)

        for i in nm.range(n):
            tmp_p[i] = 0

        for i in nm.range(n):
            for j in nm.range(m):
                b[i, j] = tmp[i, j]

    a = np.ones((n, m)).astype(dtype)
    b = np.zeros((n, m)).astype(dtype)
    copy_and_set_zero_first_col_with_empty(a, b)

    c = np.asfortranarray(a.copy())
    c[:, 0] = 0

    np.testing.assert_allclose(b, c)


def test_empty_accepts_shape_array(backend):
    n = 13
    m = 7

    @nm.jit(backend=backend)
    def copy_with_shape_array(a, b):
        shape_vec = nm.Shape(a)
        tmp = nm.empty(shape_vec, np.float64)
        tmp[:] = a
        b[:] = tmp

    a = np.random.random((n, m)).astype(np.float64)
    b = np.zeros((n, m), dtype=np.float64)
    copy_with_shape_array(a, b)
    np.testing.assert_allclose(b, a)
