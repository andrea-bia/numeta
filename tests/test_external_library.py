import numeta as nm
import numpy as np
import ctypes.util
import pytest
import os


def test_libc_getpagesize(backend):
    if ctypes.util.find_library("c") is None:
        pytest.skip("libc library not found")
    libc = nm.ExternalLibraryWrapper("c")
    libc.add_method(
        "getpagesize",
        [],
        nm.int32,
    )

    @nm.jit(backend=backend)
    def get_pagesize(pagesize):
        pagesize[:] = libc.getpagesize()

    pagesize = np.zeros((), dtype=np.int32)
    get_pagesize(pagesize)

    assert pagesize == os.sysconf("SC_PAGE_SIZE")


def test_blas(backend):
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

    @nm.jit(backend=backend)
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


def test_external_library_arg_pass_by_value_metadata():
    lib = nm.ExternalLibraryWrapper("foo")
    lib.add_method(
        "bar",
        [nm.Arg(nm.i8, pass_by_value=True), nm.Arg(nm.i8, pass_by_value=False)],
        None,
    )

    args = list(lib.methods.procedures["bar"].arguments.values())
    assert args[0].pass_by_value is True
    assert args[1].pass_by_value is False


def test_external_library_arg_pass_by_value_rejects_non_scalar():
    lib = nm.ExternalLibraryWrapper("foo")
    with pytest.raises(ValueError, match="scalar variables"):
        lib.add_method("bar", [nm.Arg(nm.f8[None], pass_by_value=True)], None)
