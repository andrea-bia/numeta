import numpy as np
import numeta as nm


def test_library_save_and_load(tmp_path):
    lib = nm.NumetaLibrary("save_and_load")

    @nm.jit(library=lib)
    def add(a):
        a[:] += 1

    array = np.zeros(4, dtype=np.int64)
    add(array)
    assert all(array == 1)

    lib.save(tmp_path)

    lib_loaded = nm.NumetaLibrary.load("save_and_load", tmp_path)

    array = np.zeros(4, dtype=np.int64)
    lib_loaded.add(array)
    assert all(array == 1)


def test_library_save_and_load_with_dep(tmp_path):
    lib = nm.NumetaLibrary("save_and_load_with_dep")

    @nm.jit(library=lib)
    def set_zero(a):
        a[:] = 0

    @nm.jit(library=lib)
    def add(a):
        set_zero(a)
        a[:] += 1

    array = np.zeros(4, dtype=np.int64)
    lib.add(array)
    assert all(array == 1)
    lib.save(tmp_path, "")
    assert len(lib._entries) == 2

    lib_loaded = nm.NumetaLibrary.load("save_and_load_with_dep", tmp_path)
    assert len(lib_loaded._entries) == 2

    array = np.zeros(4, dtype=np.int64)
    lib_loaded.add(array)
    assert all(array == 1)


def test_library_save_and_load_with_dep_2(tmp_path):
    lib = nm.NumetaLibrary("save_and_load_with_dep_2")

    @nm.jit
    def set_zero(a):
        a[:] = 0

    @nm.jit(library=lib)
    def add(a):
        set_zero(a)
        a[:] += 1

    array = np.zeros(4, dtype=np.int64)
    lib.add(array)
    assert all(array == 1)
    assert len(lib._entries) == 1
    lib.save(tmp_path, "")

    lib_loaded = nm.NumetaLibrary.load("save_and_load_with_dep_2", tmp_path)
    assert len(lib_loaded._entries) == 1

    array = np.zeros(4, dtype=np.int64)
    lib_loaded.add(array)
    assert all(array == 1)


def test_library_save_and_load_use_dep(tmp_path):
    lib = nm.NumetaLibrary("save_and_load_use_dep")

    @nm.jit(library=lib)
    def set_zero(a):
        a[:] = 0

    @nm.jit(library=lib)
    def add(a):
        set_zero(a)
        a[:] += 1

    array = np.zeros(4, dtype=np.int64)
    lib.add(array)
    assert all(array == 1)
    lib.save(tmp_path, "")

    lib_loaded = nm.NumetaLibrary.load("save_and_load_use_dep", tmp_path)

    @nm.jit
    def minus(a):
        lib_loaded.set_zero(a)
        a[:] -= 1

    array = np.zeros(4, dtype=np.int64)
    minus(array)
    assert all(array == -1)


def test_library_name_conflict(tmp_path):

    lib = nm.NumetaLibrary("name_conflict")

    @nm.jit(library=lib)
    def set_zero(a):
        a[:] = 0

    @nm.jit(library=lib)
    def add(a):
        set_zero(a)
        a[:] += 1

    array = np.zeros(4, dtype=np.int64)
    lib.add(array)
    assert all(array == 1)
    lib.save(tmp_path, "")

    lib_loaded_1 = nm.NumetaLibrary.load("name_conflict", tmp_path)
    try:
        lib_loaded_2 = nm.NumetaLibrary.load("name_conflict", tmp_path)
    except ValueError:
        pass

    array = np.zeros(4, dtype=np.int64)
    lib_loaded_1.add(array)
    assert all(array == 1)


def test_library_external_dep(tmp_path):
    import ctypes.util
    import os

    lib = nm.NumetaLibrary("external_dep")

    if ctypes.util.find_library("blas") is None:
        pytest.skip("BLAS library not found")
    blas = nm.ExternalLibraryWrapper("blas")
    blas.add_method(
        "dgemm",
        [
            nm.char,
            nm.char,
            nm.i8,
            nm.i8,
            nm.i8,
            nm.f8,
            nm.f8[None],
            nm.i8,
            nm.f8[None],
            nm.i8,
            nm.f8,
            nm.f8[None],
            nm.i8,
        ],
        None,
        bind_c=False,
    )

    n = 100

    @nm.jit(library=lib)
    def matmul(a, b, c):
        blas.dgemm(
            "N",
            "N",
            b.shape[0],
            a.shape[1],
            c.shape[1],
            1.0,
            b,
            b.shape[0],
            a,
            a.shape[0],
            0.0,
            c,
            c.shape[0],
        )

    a = np.random.rand(n, n)
    b = np.random.rand(n, n)
    c = np.zeros((n, n))

    lib.matmul(a, b, c)

    np.testing.assert_allclose(c, np.dot(a, b))
    lib.save(tmp_path)

    lib_loaded = nm.NumetaLibrary.load("external_dep", tmp_path)

    lib_loaded.matmul(a, b, c)
    np.testing.assert_allclose(c, np.dot(a, b))
