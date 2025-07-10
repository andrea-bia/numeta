import numpy as np
import numeta as nm


def test_struct_array():
    n = 2
    m = 3

    np_nested1 = np.dtype([("a", np.int64, (n, n)), ("b", np.float64, (m,))], align=True)
    np_nested2 = np.dtype([("c", np_nested1, (n,)), ("d", np_nested1, (3,))], align=True)
    np_nested3 = np.dtype([("c", np_nested2, (2,)), ("d", np_nested1, (3,))], align=True)

    @nm.jit
    def mod_struct(a) -> None:
        a[1]["c"][1]["d"][2]["b"][1] = -4.0

    a = np.zeros(2, dtype=np_nested3)

    mod_struct(a)

    b = np.zeros(2, dtype=np_nested3)
    b[1]["c"][1]["d"][2]["b"][1] = -4.0

    np.testing.assert_equal(a, b)


def test_struct():
    n = 2
    m = 3

    np_nested1 = np.dtype([("a", np.int64, (n, n)), ("b", np.float64, (m,))], align=True)
    np_nested2 = np.dtype([("c", np_nested1, (n,)), ("d", np_nested1, (3,))], align=True)
    np_nested3 = np.dtype([("c", np_nested2, (2,)), ("d", np_nested1, (3,))], align=True)

    @nm.jit
    def mod_struct(a) -> None:
        a["c"][1]["d"][2]["b"][1] = -4.0

    a = np.zeros(2, dtype=np_nested3)

    mod_struct(a[1])

    b = np.zeros(2, dtype=np_nested3)
    b[1]["c"][1]["d"][2]["b"][1] = -4.0

    np.testing.assert_equal(a, b)


def test_struct_type_helpers():
    dtype = np.dtype([("x", np.int32), ("y", np.float64)], align=True)
    st = nm.get_datatype(dtype)

    arr_t = st[2]
    assert isinstance(arr_t, nm.datatype.ArrayType)
    assert arr_t.dtype is st
    assert arr_t.shape.dims == (2,)

    @nm.jit
    def fill(out):
        tmp = st()
        tmp["x"] = 5
        tmp["y"] = 1.5
        out[0] = tmp

    out = np.zeros(1, dtype=dtype)
    fill(out)

    expected = np.zeros(1, dtype=dtype)
    expected[0]["x"] = 5
    expected[0]["y"] = 1.5
    np.testing.assert_equal(out, expected)


def test_struct_array_call():
    dtype = np.dtype([("x", np.int32), ("y", np.float64)], align=True)
    nm_dtype = nm.get_datatype(dtype)

    @nm.jit
    def fill(out):
        val = nm_dtype()
        val["x"] = 7
        val["y"] = -2.0
        out[:] = nm_dtype[2](val)

    out = np.empty(2, dtype=dtype)
    fill(out)

    expected = np.empty(2, dtype=dtype)
    expected["x"] = 7
    expected["y"] = -2.0
    np.testing.assert_equal(out, expected)


def test_struct_class_reuse():

    dtype = np.dtype([("x", np.int32), ("y", np.float64)], align=True)
    st1 = nm.get_datatype(dtype)
    st2 = nm.get_datatype(dtype)
    assert st1 is st2
