# import numpy as np
# import numeta as nm
#
#
# def test_constant_scalar(backend, backend):
#    @nm.jit(backend=backend)
#    def fill(a):
#        c = nm.constant(5)
#        a[0] = c
#
#    arr = np.zeros(1, dtype=np.int64)
#    fill(arr)
#    assert arr[0] == 5
#
#
# def test_constant_array(backend, backend):
#    @nm.jit(backend=backend)
#    def fill(a):
#        c = nm.constant([1, 2, 3])
#        for i in nm.range(3):
#            a[i] = c[i]
#
#    arr = np.zeros(3, dtype=np.int64)
#    fill(arr)
#    np.testing.assert_array_equal(arr, [1, 2, 3])
