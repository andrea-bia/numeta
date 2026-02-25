import numpy as np
import numeta as nm
import pytest


def test_declare_global_constant(backend):

    global_constant_var = nm.declare_global_constant(
        (2, 1), np.float64, value=np.array([2.0, -1.0]), name="global_constant_var", backend=backend
    )

    @nm.jit(backend=backend)
    def get_global_constant(var):
        var[0] = global_constant_var[0, 0]
        var[1] = global_constant_var[1, 0]

    a = np.empty(2, dtype=np.float64)
    get_global_constant(a)

    np.testing.assert_allclose(a, np.array([2.0, -1.0]))


def test_declare_global_constant_nested(backend):

    global_constant_var = nm.declare_global_constant(
        (2, 1), np.float64, value=np.array([2.0, -1.0]), name="global_constant_var", backend=backend
    )

    @nm.jit(backend=backend)
    def get_global_constant_nested(var):
        var[0] = global_constant_var[0, 0]
        var[1] = global_constant_var[1, 0]

    @nm.jit(backend=backend)
    def get_global_constant(var):
        get_global_constant_nested(var)

    a = np.empty(2, dtype=np.float64)
    get_global_constant(a)

    np.testing.assert_allclose(a, np.array([2.0, -1.0]))


def test_declare_global_constant_shape_vector_input(backend):

    shape_vec = nm.declare_global_constant(
        (2,),
        np.int64,
        value=np.array([2, 1], dtype=np.int64),
        name=f"global_constant_shape_vec_{backend}",
        backend=backend,
    )
    global_constant_var = nm.declare_global_constant(
        shape_vec,
        np.float64,
        value=np.array([[2.0], [-1.0]]),
        name=f"global_constant_from_shape_vec_{backend}",
        backend=backend,
    )

    @nm.jit(backend=backend)
    def get_global_constant(var):
        var[0] = global_constant_var[0, 0]
        var[1] = global_constant_var[1, 0]

    a = np.empty(2, dtype=np.float64)
    get_global_constant(a)

    np.testing.assert_allclose(a, np.array([2.0, -1.0]))


def test_declare_global_constant_shape_vector_input_nested(backend):

    shape_vec = nm.declare_global_constant(
        (2,),
        np.int64,
        value=np.array([2, 1], dtype=np.int64),
        name=f"global_constant_shape_vec_nested_{backend}",
        backend=backend,
    )
    global_constant_var = nm.declare_global_constant(
        shape_vec,
        np.float64,
        value=np.array([[2.0], [-1.0]]),
        name=f"global_constant_from_shape_vec_nested_{backend}",
        backend=backend,
    )

    @nm.jit(backend=backend)
    def get_global_constant_nested(var):
        var[0] = global_constant_var[0, 0]
        var[1] = global_constant_var[1, 0]

    @nm.jit(backend=backend)
    def get_global_constant(var):
        get_global_constant_nested(var)

    a = np.empty(2, dtype=np.float64)
    get_global_constant(a)

    np.testing.assert_allclose(a, np.array([2.0, -1.0]))


def test_declare_global_constant_shape_vector_requires_compile_time_values(backend):
    from numeta.ast import Variable

    dynamic_shape = Variable("dynamic_shape", dtype=nm.i8, shape=(2,))

    with pytest.raises(ValueError, match="compile-time integer values"):
        nm.declare_global_constant(
            dynamic_shape,
            np.float64,
            value=np.array([[2.0], [-1.0]]),
            name=f"global_constant_dynamic_shape_{backend}",
            backend=backend,
        )
