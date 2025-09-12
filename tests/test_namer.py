import numeta as nm
import numpy as np


def test_custom_namer():
    @nm.jit(namer=lambda length, *_: f"spec_{length}")
    def fill(length: nm.comptime, a):
        for i in range(length):
            a[i] = i

    arr = np.zeros(5, dtype=np.int64)
    fill(3, arr)

    assert fill.get_symbolic_functions()[0].name == "spec_3"
    np.testing.assert_allclose(arr[:3], np.arange(3))
