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

    names = [f.name for f in nm.jitted_functions()]
    assert set(names) == {"add_one", "add_two"}


def test_jitted_functions_registry_clear():
    nm.clear_jitted_functions()

    @nm.jit
    def add_one(a):
        a[:] += 1

    nm.clear_jitted_functions()

    @nm.jit
    def add_two(a):
        a[:] += 2

    array = np.zeros(10, dtype=np.int64)
    add_one(array)
    assert all(array == 1)
    add_two(array)
    assert all(array == 3)

    names = [f.name for f in nm.jitted_functions()]
    assert set(names) == {"add_two"}
