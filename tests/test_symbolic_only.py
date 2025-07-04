import numeta as nm
import numpy as np


def test_symbolic_only():
    n = 100

    @nm.jit(symbolic_only=True)
    def mul(a, b, c):
        for i in nm.range(a.shape[0]):
            for k in nm.range(b.shape[0]):
                c[i, :] += a[i, k] * b[k, :]

    a = np.random.rand(n, n)
    b = np.random.rand(n, n)
    c = np.zeros((n, n))

    sym = mul(a, b, c)

    assert isinstance(sym, nm.syntax.Subroutine)
