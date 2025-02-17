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
