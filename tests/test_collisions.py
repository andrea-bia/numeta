import numpy as np
import pytest
import numeta as nm

from numeta.numeta_function import NumetaFunction


def test_compiled_name_collision_warns():
    original_names = NumetaFunction.used_compiled_names.copy()
    NumetaFunction.used_compiled_names.clear()
    NumetaFunction.used_compiled_names.add("add_1")
    try:

        @nm.jit
        def add(a):
            a[:] += 1

        array = np.zeros(4, dtype=np.int64)
        with pytest.warns(RuntimeWarning, match="collision"):
            add(array)
    finally:
        NumetaFunction.used_compiled_names.clear()
        NumetaFunction.used_compiled_names.update(original_names)


def test_custom_namer_collision_raises():
    original_names = NumetaFunction.used_compiled_names.copy()
    NumetaFunction.used_compiled_names.clear()
    try:

        def namer(*signature):
            return "fixed_name"

        @nm.jit(namer=namer)
        def add(a):
            a[:] += 1

        array = np.zeros(4, dtype=np.int64)
        add(array)

        @nm.jit(namer=namer)
        def add2(a):
            a[:] += 1

        with pytest.raises(ValueError, match="Custom namer produced duplicate compiled name"):
            add2(array)
    finally:
        NumetaFunction.used_compiled_names.clear()
        NumetaFunction.used_compiled_names.update(original_names)
