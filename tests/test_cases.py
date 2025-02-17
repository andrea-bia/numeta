import numpy as np
import numeta as nm


def test_cases():
    @nm.jit
    def cases(a) -> None:
        a[:] = 0.0
        for i in nm.frange(n):
            for j in nm.cases(i, range(n)):
                a[j] = j

    n = 27
    a = np.empty(n, dtype=np.float64)

    cases(a)
    np.testing.assert_allclose(a, np.array(range(n), dtype=np.float64))
