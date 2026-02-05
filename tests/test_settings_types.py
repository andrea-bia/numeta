import numeta as nm
from numeta.settings import settings
import numpy as np
import pytest


@pytest.fixture
def float32_settings():
    # Set default real to float32
    settings.set_default_from_datatype(nm.float32, iso_c=True)
    yield
    # Restore default real to float64
    settings.set_default_from_datatype(nm.float64, iso_c=True)


def test_integer_input_returns_float32(float32_settings, backend):
    @nm.jit(backend=backend)
    def compute(a):
        return nm.sin(a)

    # Pass integer
    arg = 1
    res = compute(arg)

    # Verify result is float32
    # Scalar result from jit might be 0-d array or scalar
    assert res.dtype == np.float32
    np.testing.assert_allclose(res, np.sin(1.0), rtol=1e-5)


def test_integer_input_returns_float64_default(backend):
    # Ensure default is still float64 (running without fixture)
    # This assumes tests run sequentially or fixture cleans up properly

    @nm.jit(backend=backend)
    def compute(a):
        return nm.sin(a)

    arg = 1
    res = compute(arg)

    assert res.dtype == np.float64
    np.testing.assert_allclose(res, np.sin(1.0))
