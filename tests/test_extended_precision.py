import numpy as np
import pytest

import numeta as nm


HAS_FLOAT128 = hasattr(np, "float128")
HAS_COMPLEX256 = hasattr(np, "complex256")


@pytest.mark.skipif(not HAS_FLOAT128, reason="np.float128 is not available on this platform")
def test_get_datatype_float128(backend):
    assert nm.get_datatype(np.float128) is nm.float128


@pytest.mark.skipif(not HAS_COMPLEX256, reason="np.complex256 is not available on this platform")
def test_get_datatype_complex256(backend):
    assert nm.get_datatype(np.complex256) is nm.complex256


@pytest.mark.skipif(not HAS_FLOAT128, reason="np.float128 is not available on this platform")
def test_float128_aliases(backend):
    assert nm.real16 is nm.float128
    assert nm.f16 is nm.float128
    assert nm.r16 is nm.float128


@pytest.mark.skipif(not HAS_COMPLEX256, reason="np.complex256 is not available on this platform")
def test_complex256_aliases(backend):
    assert nm.complex32 is nm.complex256
    assert nm.c32 is nm.complex256


@pytest.mark.skipif(not HAS_FLOAT128, reason="np.float128 is not available on this platform")
def test_jit_float128_roundtrip(backend):
    @nm.jit(backend=backend)
    def add_one(a):
        a[:] = a[:] + np.float128(1.0)

    a = np.array([1.0, 2.0, 3.0], dtype=np.float128)
    add_one(a)
    np.testing.assert_allclose(a, np.array([2.0, 3.0, 4.0], dtype=np.float128), rtol=1e-12)


@pytest.mark.skipif(not HAS_COMPLEX256, reason="np.complex256 is not available on this platform")
def test_jit_complex256_roundtrip(backend):
    @nm.jit(backend=backend)
    def scale(a):
        a[:] = a[:] * np.complex256(2.0 - 1.0j)

    a = np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex256)
    expected = a * np.complex256(2.0 - 1.0j)
    scale(a)
    np.testing.assert_allclose(a, expected, rtol=1e-12)
