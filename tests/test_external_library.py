import numeta as nm
import numpy as np
import ctypes.util
import pytest
import os


def test_libc_getpid():
    if ctypes.util.find_library("c") is None:
        pytest.skip("libc library not found")
    libc = nm.ExternalLibraryWrapper("c")
    libc.add_method(
        "getpid",
        [],
        nm.int32,
    )

    @nm.jit
    def get_pid(pid):
        pid[:] = libc.getpid()

    pid = np.zeros((), dtype=np.int32)
    get_pid(pid)

    assert pid == os.getpid()


def test_blas():
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

    @nm.jit
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

    matmul(a, b, c)

    np.testing.assert_allclose(c, np.dot(a, b))
