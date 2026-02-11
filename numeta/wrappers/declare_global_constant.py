from __future__ import annotations
from typing import Any

import numpy as np
from numeta.ast import Variable as AstVariable, Namespace
from numeta.datatype import DataType, float64, get_datatype
from numeta.array_shape import ArrayShape
from numeta.numeta_function import NumetaCompiledFunction
from numeta.settings import settings

_n_global_constant: int = 0


def declare_global_constant(
    shape: tuple[Any, ...] | list[Any] | int | ArrayShape,
    dtype: Any = float64,
    order: str = "C",
    name: str | None = None,
    value: Any = None,
    directory: str | None = None,
    backend: str | None = None,
) -> AstVariable:
    if backend is None:
        backend = settings.default_backend
    if order not in ["C", "F"]:
        raise ValueError(f"Invalid order: {order}, must be 'C' or 'F'")

    fortran_order = order == "F"
    if not isinstance(shape, ArrayShape):
        if not isinstance(shape, (tuple, list)):
            shape = (shape,)
        shape = ArrayShape(tuple(shape), fortran_order=fortran_order)

    dtype_arg = get_datatype(dtype)

    if name is None:
        global _n_global_constant
        name = f"global_constant_{_n_global_constant}"
        _n_global_constant += 1

    # Lets create a namespace to host the global_constant variable.
    global_constant_namespace = Namespace(f"{name}_namespace")

    var = AstVariable(
        name=name,
        dtype=dtype_arg,
        shape=shape,
        assign=value,
        # TODO
        # parameter=True, # parameter is not supported yet, so not really constant.
        parent=global_constant_namespace,
    )

    # We have to compile the namespace when needed
    namespace_library = NumetaCompiledFunction(
        f"{name}_namespace",
        global_constant_namespace,
        path=directory,
        backend=backend,
    )
    global_constant_namespace.parent = namespace_library
    return var
