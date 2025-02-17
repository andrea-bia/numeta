import numeta as nm
import numpy as np
import sys


def test_lapack():
    lapack = nm.ExternalLibraryWrapper("lapack")
    lapack.add_method(
        "dgemm",
        [
            nm.char,
            nm.char,
            nm.i4,
            nm.i4,
            nm.i4,
            nm.f8,
            nm.f8[None],
            nm.i4,
            nm.f8[None],
            nm.i4,
            nm.f8,
            nm.f8[None],
            nm.i4,
        ],
        None,
        bind_c=False,
    )

    n = 100

    nm.settings.set_integer(32)

    @nm.jit
    def matmul(a, b, c):
        lapack.dgemm("N", "N", n, n, n, 1.0, b, n, a, n, 0.0, c, n)

    a = np.random.rand(n, n)
    b = np.random.rand(n, n)
    c = np.zeros((n, n))

    matmul(a, b, c)

    np.testing.assert_allclose(c, np.dot(a, b))


test_lapack()
