import numeta as nm
import numpy as np
import pytest


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_matmul_return_1_ndarray(dtype):
    n = 100

    @nm.jit
    def mul(a, b):
        c = nm.zeros((a.shape[0], b.shape[1]), dtype)
        for i in nm.range(a.shape[0]):
            for k in nm.range(b.shape[0]):
                c[i, :] += a[i, k] * b[k, :]
        return c

    a = np.random.rand(n, n).astype(dtype)
    b = np.random.rand(n, n).astype(dtype)

    c = mul(a, b)

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_allclose(c, np.dot(a, b), atol=0)
    else:
        np.testing.assert_allclose(c, np.dot(a, b), rtol=10e2 * np.finfo(dtype).eps)


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_matmul_return_2_ndarray(dtype):
    n = 100

    @nm.jit
    def mul(a, b):
        c = nm.zeros((a.shape[0], b.shape[1]), dtype)
        d = nm.zeros((a.shape[0], b.shape[1]), dtype)
        for i in nm.range(a.shape[0]):
            for k in nm.range(b.shape[0]):
                c[i, :] += a[i, k] * b[k, :]
        return c, d

    a = np.random.rand(n, n).astype(dtype)
    b = np.random.rand(n, n).astype(dtype)

    c, d = mul(a, b)

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_allclose(c, np.dot(a, b), atol=0)
        np.testing.assert_allclose(d, np.zeros((n, n), dtype=dtype), atol=0)
    else:
        np.testing.assert_allclose(c, np.dot(a, b), rtol=10e2 * np.finfo(dtype).eps)
        np.testing.assert_allclose(
            d, np.zeros((n, n), dtype=dtype), rtol=10e2 * np.finfo(dtype).eps
        )


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_return_scalar(dtype):

    @nm.jit
    def return_scalar():
        return nm.scalar(dtype, 42)

    scalar = return_scalar()

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_allclose(scalar, 42, atol=0)
    else:
        np.testing.assert_allclose(scalar, 42, rtol=10e2 * np.finfo(dtype).eps)


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_return_1_ndarray_getitem(dtype):
    n = 100
    m = 50

    @nm.jit
    def mul(a, b):
        c = nm.zeros((a.shape[0], b.shape[1]), dtype)
        for i in nm.range(a.shape[0]):
            for k in nm.range(b.shape[0]):
                c[i, :] += a[i, k] * b[k, :]
        return c[:10, : n // 2]

    a = np.random.rand(n, m).astype(dtype)
    b = np.random.rand(m, n).astype(dtype)

    c = mul(a, b)

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_allclose(c, np.dot(a, b)[:10, : n // 2], atol=0)
    else:
        np.testing.assert_allclose(c, np.dot(a, b)[:10, : n // 2], rtol=10e2 * np.finfo(dtype).eps)


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_return_1_ndarray_sum(dtype):
    n = 100
    m = 50

    @nm.jit
    def return_1_ndarray_sum(a, b):
        return a + b

    a = np.random.rand(n, m).astype(dtype)
    b = np.random.rand(n, m).astype(dtype)

    c = return_1_ndarray_sum(a, b)

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_allclose(c, a + b, atol=0)
    else:
        np.testing.assert_allclose(c, a + b, rtol=10e2 * np.finfo(dtype).eps)


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_return_1_ndarray_transpose(dtype):
    shape = (18, 7)

    @nm.jit
    def transpose_expr(a):
        return nm.transpose(a)

    a = np.random.random(shape).astype(dtype)

    out = transpose_expr(a)
    expected = np.transpose(a)

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_array_equal(out, expected)
    else:
        np.testing.assert_allclose(out, expected, rtol=10e2 * np.finfo(dtype).eps)


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_return_1_ndarray_mixed_expression(dtype):
    shape = (24, 16)

    @nm.jit
    def combined_expression(a, b):
        rows = a.shape[0] // 2
        cols = a.shape[1] // 2
        left = a[:rows, :cols] + b[:rows, :cols]
        right = a[:rows, :cols] - b[:rows, :cols]
        return left * right

    a = np.random.random(shape).astype(dtype)
    b = np.random.random(shape).astype(dtype)

    out = combined_expression(a, b)
    rows = shape[0] // 2
    cols = shape[1] // 2
    expected = (a[:rows, :cols] + b[:rows, :cols]) * (a[:rows, :cols] - b[:rows, :cols])

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_array_equal(out, expected)
    else:
        np.testing.assert_allclose(out, expected, rtol=10e2 * np.finfo(dtype).eps)


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_call_returning_function_scalar(dtype):

    @nm.jit
    def return_scalar():
        return nm.scalar(dtype, 42)

    @nm.jit
    def caller():
        return return_scalar()

    scalar = caller()

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_allclose(scalar, 42, atol=0)
    else:
        np.testing.assert_allclose(scalar, 42, rtol=10e2 * np.finfo(dtype).eps)


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_call_returning_function_array_rank1(dtype):

    size = 40

    @nm.jit
    def return_array_rank1():
        a = nm.zeros(size, dtype)
        return a

    @nm.jit
    def caller():
        return return_array_rank1()

    array = caller()

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_allclose(array, np.zeros(size, dtype=dtype), atol=0)
    else:
        np.testing.assert_allclose(
            array, np.zeros(size, dtype=dtype), rtol=10e2 * np.finfo(dtype).eps
        )


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_call_returning_function_array_rank2(dtype):

    @nm.jit
    def return_array_rank2(n, m):
        a = nm.zeros((n, m), dtype)
        return a

    @nm.jit
    def caller_rank2(n, m):
        return return_array_rank2(n, m)

    size = (10, 20)
    array = caller_rank2(*size)

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_allclose(array, np.zeros(size, dtype=dtype), atol=0)
    else:
        np.testing.assert_allclose(
            array, np.zeros(size, dtype=dtype), rtol=10e2 * np.finfo(dtype).eps
        )


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_nested_return_array_rank1(dtype):

    size = 40

    @nm.jit
    def return_array_rank1():
        a = nm.zeros(size, dtype)
        return a

    @nm.jit
    def caller(b):
        b[:] = return_array_rank1()

    array = np.ones(size, dtype=dtype)
    caller(array)

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_allclose(array, np.zeros(size, dtype=dtype), atol=0)
    else:
        np.testing.assert_allclose(
            array, np.zeros(size, dtype=dtype), rtol=10e2 * np.finfo(dtype).eps
        )


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_nested_return_array_rank1_no_numpy_allocator(dtype):

    size = 40

    nm.settings.unset_numpy_allocator()

    @nm.jit
    def return_array_rank1():
        a = nm.zeros(size, dtype)
        return a

    @nm.jit
    def caller(b):
        b[:] = return_array_rank1()

    array = np.ones(size, dtype=dtype)
    caller(array)

    nm.settings.set_numpy_allocator()

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_allclose(array, np.zeros(size, dtype=dtype), atol=0)
    else:
        np.testing.assert_allclose(
            array, np.zeros(size, dtype=dtype), rtol=10e2 * np.finfo(dtype).eps
        )


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_inline_returning_function_scalar(dtype):
    @nm.jit(inline=True)
    def return_scalar_inline():
        return nm.scalar(dtype, 7)

    @nm.jit
    def caller_scalar():
        return return_scalar_inline()

    scalar = caller_scalar()

    expected_scalar = dtype(7)

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_allclose(scalar, expected_scalar, atol=0)
    else:
        np.testing.assert_allclose(scalar, expected_scalar, rtol=10e2 * np.finfo(dtype).eps)


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_inline_returning_function_array(dtype):
    size = 8

    @nm.jit(inline=True)
    def return_array_inline(n):
        out = nm.zeros(n, dtype)
        out[:] = nm.scalar(dtype, 3)
        return out

    @nm.jit
    def caller_array(n):
        return return_array_inline(n)

    array = caller_array(size)

    expected_array = np.full(size, dtype(3), dtype=dtype)

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_allclose(array, expected_array, atol=0)
    else:
        np.testing.assert_allclose(array, expected_array, rtol=10e2 * np.finfo(dtype).eps)
