import numpy as np
import pytest
import numeta as nm


@pytest.mark.parametrize(
    "dtype",
    [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128],
)
def test_scalar_call(dtype):

    @nm.jit
    def fill(dtype: nm.comptime, a):
        nm_dtype = nm.get_datatype(dtype)

        a[0] = nm_dtype(50)

        tmp2 = nm_dtype(100)
        a[1] = tmp2

    a = np.empty(2, dtype=dtype)
    fill(dtype, a)

    expected = np.empty(2, dtype=dtype)
    expected[0] = 50.0
    expected[1] = 100.0

    np.testing.assert_allclose(a, expected)


@pytest.mark.parametrize(
    "dtype",
    [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128],
)
def test_array_call_matrix(dtype):

    @nm.jit
    def fill(a):
        nm_dtype = nm.get_datatype(dtype)
        a[:2, :2] = nm_dtype[2, 3](50)[:2, :2]

        tmp2 = nm_dtype[2](100)
        a[:, 2] = tmp2[:2]

    out = np.empty((2, 3), dtype=dtype)
    fill(out)

    expected = np.empty((2, 3), dtype=dtype)
    expected[:2, :2] = 50
    expected[:, 2] = 100

    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize(
    "dtype",
    [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128],
)
def test_array_call_3d(dtype):

    @nm.jit
    def fill(a):
        nm_dtype = nm.get_datatype(dtype)
        a[:] = nm_dtype[2, 2, 3](7)

    out = np.empty((2, 2, 3), dtype=dtype)
    fill(out)

    expected = np.full((2, 2, 3), 7, dtype=dtype)
    np.testing.assert_allclose(out, expected)


def test_variable_declaration_syntax():

    a = nm.float64(name="a")
    from numeta.syntax.statements import VariableDeclaration

    dec = VariableDeclaration(a)
    assert dec.print_lines() == ["real(c_double) :: a\n"]

    array = nm.int32[10, 20](name="array")
    dec = VariableDeclaration(array)
    assert dec.print_lines() == ["integer(c_int32_t), dimension(0:19, 0:9) :: array\n"]

    array_unknown = nm.int32[None](name="array_unknown")
    dec = VariableDeclaration(array_unknown)
    print(dec.print_lines())


test_variable_declaration_syntax()
