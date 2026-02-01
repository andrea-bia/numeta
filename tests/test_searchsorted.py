# import numeta as nm
# import numpy as np
#
#
# def test_searchsorted_left(backend, backend):
#    a = np.arange(100, dtype=np.int64)
#    v = np.int64(42)
#    idx = nm.searchsorted(a, v)
#    expected = np.searchsorted(a, v)
#    np.testing.assert_equal(idx, expected)
#
#
# def test_searchsorted_right(backend, backend):
#    a = np.array([1, 1, 2, 2, 3, 3], dtype=np.int64)
#    v = np.int64(2)
#    idx = nm.searchsorted(a, v, side="right")
#    expected = np.searchsorted(a, v, side="right")
#    np.testing.assert_equal(idx, expected)
