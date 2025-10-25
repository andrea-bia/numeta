import numpy as np
import numeta as nm


def test_jitted_functions_registry():
    nm.clear_jitted_functions()

    @nm.jit
    def add_one(a):
        a[:] += 1

    @nm.jit
    def add_two(a):
        a[:] += 2

    array = np.zeros(10, dtype=np.int64)
    add_one(array)
    assert all(array == 1)
    add_two(array)
    assert all(array == 3)

    assert len(nm.jitted_functions()) == 2


def test_jitted_functions_registry_clear():
    nm.clear_jitted_functions()

    @nm.jit
    def add_one(a):
        a[:] += 1

    @nm.jit
    def add_two(a):
        a[:] += 2

    array = np.zeros(10, dtype=np.int64)
    add_one(array)
    assert all(array == 1)
    nm.clear_jitted_functions()

    add_two(array)
    assert all(array == 3)

    assert len(nm.jitted_functions()) == 1
