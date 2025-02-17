import numeta as nm
import numpy as np


def test_omp():
    @nm.jit
    def test_mul(a, b, c):
        n_threads = nm.scalar(nm.i8, nm.omp.omp_get_max_threads())
        i_thread = nm.scalar(nm.i8, 0)

        for j in nm.prange(
            b.shape[1],
            shared=[a, b, c, a.shape[0].variable, b.shape[0].variable],
            schedule="static",
        ):
            for k in nm.frange(b.shape[0]):
                for i in nm.frange(a.shape[0]):
                    i_thread[:] = nm.omp.omp_get_thread_num()

                    nm.omp.atomic_update_add(c[i, j].real, a[i, k].real * b[k, j].real)
                    nm.omp.atomic_update_sub(c[i, j].real, a[i, k].imag * b[k, j].imag)
                    nm.omp.atomic_update_add(c[i, j].imag, a[i, k].real * b[k, j].imag)
                    nm.omp.atomic_update_add(c[i, j].imag, a[i, k].imag * b[k, j].real)

    n = 50

    a = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    b = np.random.rand(n, n) + 1j * np.random.rand(n, n)

    c = np.zeros((n, n), dtype=np.complex128)

    test_mul(a, b, c)

    np.testing.assert_allclose(c, a.dot(b))


test_omp()
