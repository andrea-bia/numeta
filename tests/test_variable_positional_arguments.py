import numpy as np
import pytest
import numeta as nm


@pytest.mark.parametrize("n_args", range(1, 5))
@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
@pytest.mark.parametrize("shape", [(), (5,), (2, 3)])
def test_variable_number_of_arguments(n_args, dtype, shape):
    @nm.jit
    def fill(*args):
        for i, arg in enumerate(args):
            arg[:] = float(i)

    args = [np.empty(shape, dtype=dtype) for _ in range(n_args)]
    fill(*args)

    for i, arg in enumerate(args):
        expected = np.full(shape, float(i), dtype=dtype)
        np.testing.assert_allclose(arg, expected)


def test_no_args():
    @nm.jit
    def no_op(*args):
        pass

    no_op()
