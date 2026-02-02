import numpy as np
import sys
from pathlib import Path

import pytest
import numeta as nm

from numeta.pyc_extension import PyCExtension


def test_library_save_and_load(tmp_path, backend):
    lib = nm.NumetaLibrary(f"save_and_load_{backend}")

    @nm.jit(backend=backend, library=lib)
    def add(a):
        a[:] += 1

    array = np.zeros(4, dtype=np.int64)
    add(array)
    assert all(array == 1)


def test_library_write_code(tmp_path, backend):
    lib = nm.NumetaLibrary(f"write_code_{backend}")

    @nm.jit(backend=backend, library=lib)
    def add(a):
        a[:] += 1

    @nm.jit(backend=backend, library=lib)
    def mul(a):
        a[:] *= 2

    array = np.ones(4, dtype=np.int64)
    lib.add(array)
    lib.mul(array)

    lib.write_code(tmp_path)

    compiled_names = []
    for nm_function in lib._entries.values():
        compiled_names.extend(
            [compiled.name for compiled in nm_function._compiled_functions.values()]
        )

    for name in compiled_names:
        if backend == "fortran":
            src = Path(tmp_path) / f"{name}_src.f90"
            assert src.exists()
            code = src.read_text().lower()
            assert f"subroutine {name}" in code
        elif backend == "c":
            src = Path(tmp_path) / f"{name}_src.c"
            assert src.exists()
            code = src.read_text().lower()
            assert f"void {name}" in code
        else:
            raise ValueError(f"Unsupported backend: {backend}")


def test_library_save_and_load_with_dep(tmp_path, backend):
    lib = nm.NumetaLibrary(f"save_and_load_with_dep_{backend}")

    @nm.jit(backend=backend, library=lib)
    def set_zero(a):
        a[:] = 0

    @nm.jit(backend=backend, library=lib)
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
    lib_loaded = nm.NumetaLibrary.load(f"save_and_load_with_dep_{backend}", tmp_path)
    assert len(lib_loaded._entries) == 2

    array = np.zeros(4, dtype=np.int64)
    lib_loaded.add(array)
    assert all(array == 1)


def test_library_save_and_load_with_dep_2(tmp_path, backend):
    lib = nm.NumetaLibrary(f"save_and_load_with_dep_2_{backend}")

    @nm.jit(backend=backend)
    def set_zero(a):
        a[:] = 0

    @nm.jit(backend=backend, library=lib)
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
    lib_loaded = nm.NumetaLibrary.load(f"save_and_load_with_dep_2_{backend}", tmp_path)
    assert len(lib_loaded._entries) == 1

    array = np.zeros(4, dtype=np.int64)
    lib_loaded.add(array)
    assert all(array == 1)


def test_library_save_and_load_use_dep(tmp_path, backend):
    lib = nm.NumetaLibrary(f"save_and_load_use_dep_{backend}")

    @nm.jit(backend=backend, library=lib)
    def set_zero(a):
        a[:] = 0

    @nm.jit(backend=backend, library=lib)
    def add(a):
        set_zero(a)
        a[:] += 1

    array = np.zeros(4, dtype=np.int64)
    lib.add(array)
    assert all(array == 1)
    lib.save(tmp_path, "")

    set_zero.clear()
    add.clear()
    lib_loaded = nm.NumetaLibrary.load(f"save_and_load_use_dep_{backend}", tmp_path)

    @nm.jit(backend=backend)
    def minus(a):
        lib_loaded.set_zero(a)
        a[:] -= 1

    array = np.zeros(4, dtype=np.int64)
    minus(array)
    assert all(array == -1)


def test_library_global_variable_dep(tmp_path, backend):
    lib = nm.NumetaLibrary(f"global_variable_dep_{backend}")

    global_constant_var = nm.declare_global_constant(
        (2, 1), np.float64, value=np.array([2.0, -1.0]), name="global_constant_var"
    )

    @nm.jit(backend=backend, library=lib)
    def set(var):
        var[0] = global_constant_var[0, 0]
        var[1] = global_constant_var[1, 0]

    a = np.empty(2, dtype=np.float64)
    lib.set(a)
    np.testing.assert_allclose(a, np.array([2.0, -1.0]))

    lib.save(tmp_path, "")

    set.clear()
    lib_loaded = nm.NumetaLibrary.load(f"global_variable_dep_{backend}", tmp_path)

    a = np.empty(2, dtype=np.float64)
    lib.set(a)
    np.testing.assert_allclose(a, np.array([2.0, -1.0]))


def test_library_name_conflict(tmp_path, backend):

    lib = nm.NumetaLibrary(f"name_conflict_{backend}")

    @nm.jit(backend=backend, library=lib)
    def set_zero(a):
        a[:] = 0

    @nm.jit(backend=backend, library=lib)
    def add(a):
        set_zero(a)
        a[:] += 1

    array = np.zeros(4, dtype=np.int64)
    lib.add(array)
    assert all(array == 1)
    lib.save(tmp_path, "")

    set_zero.clear()
    add.clear()
    lib_loaded_1 = nm.NumetaLibrary.load(f"name_conflict_{backend}", tmp_path)
    try:
        lib_loaded_2 = nm.NumetaLibrary.load(f"name_conflict_{backend}", tmp_path)
    except ValueError:
        pass

    array = np.zeros(4, dtype=np.int64)
    lib_loaded_1.add(array)
    assert all(array == 1)


def test_library_external_dep(tmp_path, backend):
    import ctypes.util
    import os

    lib = nm.NumetaLibrary(f"external_dep_{backend}")

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

    @nm.jit(backend=backend, library=lib)
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
    lib_loaded = nm.NumetaLibrary.load(f"external_dep_{backend}", tmp_path)

    lib_loaded.matmul(a, b, c)
    np.testing.assert_allclose(c, np.dot(a, b))


def test_library_public_api(backend):
    lib = nm.NumetaLibrary(f"public_api_{backend}")

    @nm.jit(backend=backend)
    def add(a):
        a[:] += 1

    registered = lib.register(add)
    assert registered is add
    assert "add" in lib
    assert lib["add"] is add
    assert lib.list_functions() == ["add"]
    assert len(lib) == 1
    assert list(lib)[0] is add

    lib.remove("add")
    assert "add" not in lib
    assert len(lib) == 0


def test_library_register_rejects_reserved_name(backend):
    lib = nm.NumetaLibrary(f"public_api_reserved_{backend}")

    with pytest.raises(ValueError, match="reserved"):

        @nm.jit(backend=backend, library=lib)
        def register(a):
            a[:] += 1


def test_library_load_collision_warns(tmp_path, backend):
    from numeta.numeta_function import NumetaFunction

    original_names = NumetaFunction.used_compiled_names.copy()
    NumetaFunction.used_compiled_names.clear()
    try:
        lib_dir = tmp_path / f"collision_lib_{backend}"
        lib_dir.mkdir()
        lib = nm.NumetaLibrary(f"collision_lib_{backend}")

        @nm.jit(backend=backend, library=lib)
        def add(a):
            a[:] += 1

        array = np.zeros(4, dtype=np.int64)
        lib.add(array)
        lib.save(lib_dir)

        NumetaFunction.used_compiled_names.clear()

        @nm.jit(backend=backend)
        def add(a):
            a[:] += 1

        array = np.zeros(4, dtype=np.int64)
        add(array)

        with pytest.warns(RuntimeWarning, match="collision"):
            lib_loaded = nm.NumetaLibrary.load(f"collision_lib_{backend}", lib_dir)

    finally:
        NumetaFunction.used_compiled_names.clear()
        NumetaFunction.used_compiled_names.update(original_names)


def test_library_reserved_suffix_rejected(tmp_path, backend):
    reserved_name = f"bad{PyCExtension.SUFFIX}"
    with pytest.raises(ValueError, match="reserved"):
        nm.NumetaLibrary(reserved_name)
    with pytest.raises(ValueError, match="reserved"):
        nm.NumetaLibrary.load(reserved_name, tmp_path)


def test_library_wrapper_module_collision(backend):
    wrapper_module = f"collision{PyCExtension.SUFFIX}"
    sys.modules[wrapper_module] = object()
    try:
        with pytest.raises(ValueError, match="wrapper module"):
            nm.NumetaLibrary("collision")
    finally:
        sys.modules.pop(wrapper_module, None)
