import numeta as nm
import numpy as np
import pytest


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_cast(dtype):

    @nm.jit
    def set_nine(a):
        a_int = nm.cast(a, dtype)
        a_int[:] = 9.0

    # should contain everything
    a = np.ones(16, dtype=np.bool_)
    set_nine(a)

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_allclose(a.view(dtype)[0], np.array(9, dtype=dtype), atol=0)
    else:
        np.testing.assert_allclose(
            a.view(dtype)[0], np.array(9, dtype=dtype), rtol=10e2 * np.finfo(dtype).eps
        )


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
def test_cast_getitem(dtype):

    @nm.jit
    def set_nine(a):
        a_int = nm.cast(a[0], dtype)
        a_int[:] = 9.0

    # should contain everything
    a = np.ones(16, dtype=np.bool_)
    set_nine(a)

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_allclose(a.view(dtype)[0], np.array(9, dtype=dtype), atol=0)
    else:
        np.testing.assert_allclose(
            a.view(dtype)[0], np.array(9, dtype=dtype), rtol=10e2 * np.finfo(dtype).eps
        )


def test_cast_struct():

    dtype = np.dtype([("a", np.float64), ("b", np.int64)])

    @nm.jit
    def set_nine(a):
        a_int = nm.cast(a, dtype)
        a_int["a"][:] = 9.0
        a_int["b"][:] = 9

    # should contain everything
    a = np.ones(16, dtype=np.bool_)
    set_nine(a)

    check = np.empty((), dtype)
    check["a"] = 9.0
    check["b"] = 9

    print(a.view(dtype)["a"][0], check["a"])
    print(a.view(dtype)["b"][0], check["b"])

    # cannot use rtol
    np.testing.assert_allclose(a.view(dtype)["a"][0], check["a"], atol=0)
    np.testing.assert_allclose(a.view(dtype)["b"][0], check["b"], atol=0)
