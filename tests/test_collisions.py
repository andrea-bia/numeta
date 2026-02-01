import numpy as np
import pytest
import numeta as nm

from numeta.numeta_function import NumetaFunction
from numeta.numeta_library import NumetaLibrary
from numeta.pyc_extension import PyCExtension


def test_compiled_name_collision_warns(backend, backend):
    original_names = NumetaFunction.used_compiled_names.copy()
    NumetaFunction.used_compiled_names.clear()
    NumetaFunction.used_compiled_names.add("add_1")
    try:

        @nm.jit(backend=backend)
        def add(a):
            a[:] += 1

        array = np.zeros(4, dtype=np.int64)
        with pytest.warns(RuntimeWarning, match="collision"):
            add(array)
    finally:
        NumetaFunction.used_compiled_names.clear()
        NumetaFunction.used_compiled_names.update(original_names)


def test_custom_namer_collision_raises(backend, backend):
    original_names = NumetaFunction.used_compiled_names.copy()
    NumetaFunction.used_compiled_names.clear()
    try:

        def namer(*signature):
            return "fixed_name"

        @nm.jit(backend=backend, namer=namer)
        def add(a):
            a[:] += 1

        array = np.zeros(4, dtype=np.int64)
        add(array)

        @nm.jit(backend=backend, namer=namer)
        def add2(a):
            a[:] += 1

        with pytest.raises(ValueError, match="Custom namer produced duplicate compiled name"):
            add2(array)
    finally:
        NumetaFunction.used_compiled_names.clear()
        NumetaFunction.used_compiled_names.update(original_names)


def test_compiled_name_reserved_suffix_raises(backend, backend):
    original_names = NumetaFunction.used_compiled_names.copy()
    NumetaFunction.used_compiled_names.clear()
    try:

        def namer(*signature):
            return f"fixed{PyCExtension.SUFFIX}"

        @nm.jit(backend=backend, namer=namer)
        def add(a):
            a[:] += 1

        array = np.zeros(4, dtype=np.int64)
        with pytest.raises(ValueError, match="reserved"):
            add(array)
    finally:
        NumetaFunction.used_compiled_names.clear()
        NumetaFunction.used_compiled_names.update(original_names)


def test_compiled_name_loaded_library_collision(backend, backend):
    original_names = NumetaFunction.used_compiled_names.copy()
    original_loaded = NumetaLibrary.loaded.copy()
    NumetaFunction.used_compiled_names.clear()
    NumetaLibrary.loaded.clear()
    NumetaLibrary.loaded.add("add_0")
    try:

        @nm.jit(backend=backend)
        def add(a):
            a[:] += 1

        array = np.zeros(4, dtype=np.int64)
        with pytest.raises(ValueError, match="NumetaLibrary"):
            add(array)
    finally:
        NumetaFunction.used_compiled_names.clear()
        NumetaFunction.used_compiled_names.update(original_names)
        NumetaLibrary.loaded.clear()
        NumetaLibrary.loaded.update(original_loaded)
