import pytest
import sys
from numeta.settings import settings
from numeta.numeta_function import NumetaFunction, _c_dispatch_base_available


def test_use_c_dispatch_setting():
    # Only run this test if C extension is available, otherwise uses_c_dispatch is always False
    if not _c_dispatch_base_available:
        pytest.skip("C extension not available, cannot test toggling")

    # Store original value
    orig = settings.use_c_dispatch

    try:
        # 1. Enable C dispatch
        settings.use_c_dispatch = True

        def f1(x):
            return x

        nf1 = NumetaFunction(f1)

        # Should be True
        assert nf1.uses_c_dispatch is True

        # 2. Disable C dispatch
        settings.use_c_dispatch = False

        def f2(x):
            return x

        nf2 = NumetaFunction(f2)

        # Should be False
        assert nf2.uses_c_dispatch is False

        # Verify call works (via Python dispatch)
        # This exercises the _python_call path (delegated from C or aliased)
        # Since _c_dispatch_base_available is True, NumetaFunction inherits C BaseFunction.
        # But uses_c_dispatch is False, so custom parser is skipped.
        # And C BaseFunction_call checks settings.use_c_dispatch (which is False),
        # so it delegates to _python_call.
        res = nf2(10)
        assert res == 10

    finally:
        settings.use_c_dispatch = orig
