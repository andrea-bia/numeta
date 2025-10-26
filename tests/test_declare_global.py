import numpy as np
import numeta as nm


def test_declare_global_constant():

    global_constant_var = nm.declare_global_constant(
        (2, 1), np.float64, value=np.array([2.0, -1.0]), name="global_constant_var"
    )

    @nm.jit
    def get_global_constant(var):
        var[0] = global_constant_var[0, 0]
        var[1] = global_constant_var[1, 0]

    a = np.empty(2, dtype=np.float64)
    get_global_constant(a)

    np.testing.assert_allclose(a, np.array([2.0, -1.0]))


def test_declare_global_constant_nested():

    global_constant_var = nm.declare_global_constant(
        (2, 1), np.float64, value=np.array([2.0, -1.0]), name="global_constant_var"
    )

    @nm.jit
    def get_global_constant_nested(var):
        var[0] = global_constant_var[0, 0]
        var[1] = global_constant_var[1, 0]

    @nm.jit
    def get_global_constant(var):
        get_global_constant_nested(var)

    a = np.empty(2, dtype=np.float64)
    get_global_constant(a)

    np.testing.assert_allclose(a, np.array([2.0, -1.0]))
