import os

import numeta as nm
import pytest


_BACKEND = os.getenv("NUMETA_BACKEND")
_BACKEND_PARAMS = [_BACKEND] if _BACKEND else ["fortran", "c"]


@pytest.fixture(params=_BACKEND_PARAMS, autouse=True)
def backend(request, monkeypatch):
    """
    Parametrized fixture that runs tests on both 'fortran' and 'c' backends.
    It monkeypatches numeta.jit to use the current backend by default,
    but tests can also request the 'backend' argument explicitly.
    """
    current_backend = request.param
    original_jit = nm.jit

    def jit_wrapper(*args, **kwargs):
        if "backend" not in kwargs:
            kwargs["backend"] = current_backend
        return original_jit(*args, **kwargs)

    monkeypatch.setattr(nm, "jit", jit_wrapper)
    return current_backend
