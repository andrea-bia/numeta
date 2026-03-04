import pytest

import numeta as nm
from numeta.ast.procedure import Procedure
from numeta.numeta_function import _c_dispatch_base_available
from numeta.settings import settings


def _assert_codegen_only_target(nm_function):
    assert len(nm_function._compiled_functions) == 1
    signature, target = next(iter(nm_function._compiled_functions.items()))
    assert target.compiled is False
    assert signature not in nm_function._pyc_extensions
    return signature, target


@pytest.mark.parametrize("use_c_dispatch", [False, True])
def test_arraytype_call_generates_symbolic_only(backend, use_c_dispatch):
    if use_c_dispatch and not _c_dispatch_base_available:
        pytest.skip("C dispatch extension not available")

    original_use_c_dispatch = settings.use_c_dispatch
    original_use_c_signature_parser = settings.use_c_signature_parser
    settings.use_c_dispatch = use_c_dispatch
    settings.use_c_signature_parser = True

    try:

        @nm.jit(backend=backend)
        def add_one(a):
            return a + 1

        result = add_one(nm.float64[3])
        assert isinstance(result, Procedure)

        signature, _target = _assert_codegen_only_target(add_one)
        assert signature == (("a", nm.float64.get_numpy(), 1, False, "inout", (3,)),)
    finally:
        settings.use_c_dispatch = original_use_c_dispatch
        settings.use_c_signature_parser = original_use_c_signature_parser


@pytest.mark.parametrize("use_c_dispatch", [False, True])
def test_datatype_call_generates_symbolic_only(backend, use_c_dispatch):
    if use_c_dispatch and not _c_dispatch_base_available:
        pytest.skip("C dispatch extension not available")

    original_use_c_dispatch = settings.use_c_dispatch
    original_use_c_signature_parser = settings.use_c_signature_parser
    settings.use_c_dispatch = use_c_dispatch
    settings.use_c_signature_parser = True

    try:

        @nm.jit(backend=backend)
        def add_one(a):
            return a + 1

        result = add_one(nm.float64)
        assert isinstance(result, Procedure)

        signature, _target = _assert_codegen_only_target(add_one)
        assert signature == (("a", nm.float64.get_numpy()),)
    finally:
        settings.use_c_dispatch = original_use_c_dispatch
        settings.use_c_signature_parser = original_use_c_signature_parser
