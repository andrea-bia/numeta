import numpy as np
import sys

import pytest
import numeta as nm

from numeta.pyc_extension import PyCExtension


def test_library_save_and_load(tmp_path):
    lib = nm.NumetaLibrary("save_and_load")

    @nm.jit(library=lib)
    def add(a):
        a[:] += 1

    array = np.zeros(4, dtype=np.int64)
    add(array)
    assert all(array == 1)

    lib.save(tmp_path)

    # clear add otherwise cannot load it again
    add.clear()
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

    set_zero.clear()
    add.clear()
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

    set_zero.clear()
    add.clear()
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

    set_zero.clear()
    add.clear()
    lib_loaded = nm.NumetaLibrary.load("save_and_load_use_dep", tmp_path)

    @nm.jit
    def minus(a):
        lib_loaded.set_zero(a)
        a[:] -= 1

    array = np.zeros(4, dtype=np.int64)
    minus(array)
    assert all(array == -1)


def test_library_global_variable_dep(tmp_path):
    lib = nm.NumetaLibrary("global_variable_dep")

    global_constant_var = nm.declare_global_constant(
        (2, 1), np.float64, value=np.array([2.0, -1.0]), name="global_constant_var"
    )

    @nm.jit(library=lib)
    def set(var):
        var[0] = global_constant_var[0, 0]
        var[1] = global_constant_var[1, 0]

    a = np.empty(2, dtype=np.float64)
    lib.set(a)
    np.testing.assert_allclose(a, np.array([2.0, -1.0]))

    lib.save(tmp_path, "")

    set.clear()
    lib_loaded = nm.NumetaLibrary.load("global_variable_dep", tmp_path)

    a = np.empty(2, dtype=np.float64)
    lib.set(a)
    np.testing.assert_allclose(a, np.array([2.0, -1.0]))


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

    set_zero.clear()
    add.clear()
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

    matmul.clear()
    lib_loaded = nm.NumetaLibrary.load("external_dep", tmp_path)

    lib_loaded.matmul(a, b, c)
    np.testing.assert_allclose(c, np.dot(a, b))


def test_library_load_collision_warns(tmp_path):
    from numeta.numeta_function import NumetaFunction

    original_names = NumetaFunction.used_compiled_names.copy()
    NumetaFunction.used_compiled_names.clear()
    try:
        lib_dir = tmp_path / "collision_lib"
        lib_dir.mkdir()
        lib = nm.NumetaLibrary("collision_lib")

        @nm.jit(library=lib)
        def add(a):
            a[:] += 1

        array = np.zeros(4, dtype=np.int64)
        lib.add(array)
        lib.save(lib_dir)

        NumetaFunction.used_compiled_names.clear()

        @nm.jit
        def add(a):
            a[:] += 1

        array = np.zeros(4, dtype=np.int64)
        add(array)

        with pytest.warns(RuntimeWarning, match="collision"):
            lib_loaded = nm.NumetaLibrary.load("collision_lib", lib_dir)

    finally:
        NumetaFunction.used_compiled_names.clear()
        NumetaFunction.used_compiled_names.update(original_names)


def test_library_reserved_suffix_rejected(tmp_path):
    reserved_name = f"bad{PyCExtension.SUFFIX}"
    with pytest.raises(ValueError, match="reserved"):
        nm.NumetaLibrary(reserved_name)
    with pytest.raises(ValueError, match="reserved"):
        nm.NumetaLibrary.load(reserved_name, tmp_path)


def test_library_wrapper_module_collision():
    wrapper_module = f"collision{PyCExtension.SUFFIX}"
    sys.modules[wrapper_module] = object()
    try:
        with pytest.raises(ValueError, match="wrapper module"):
            nm.NumetaLibrary("collision")
    finally:
        sys.modules.pop(wrapper_module, None)
