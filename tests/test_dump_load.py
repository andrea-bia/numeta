import numeta as nm
import numpy as np
import tempfile
import shutil


def test_dump_load():
    n = 100

    a = np.random.rand(n, n)

    dumpdir = tempfile.mkdtemp()
    tmpdir = tempfile.mkdtemp()

    @nm.jit(directory=tmpdir)
    def fill(value: nm.comptime, a):
        a[:] = value

    fill(1.0, a)

    fill.dump(dumpdir)

    shutil.rmtree(tmpdir)

    @nm.jit
    def fill(value: nm.comptime, a):
        raise Warning("This should not be called")

    fill.load(dumpdir)

    fill(1.0, a)

    np.testing.assert_allclose(a, np.ones((n, n)), rtol=10e2 * np.finfo(np.float64).eps)
