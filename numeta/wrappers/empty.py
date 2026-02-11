from __future__ import annotations
from typing import Any

from numeta.builder_helper import BuilderHelper
from numeta.datatype import DataType, float64, get_datatype
from numeta.array_shape import ArrayShape
from numeta.ast.variable import Variable


def empty(
    shape: tuple[Any, ...] | list[Any] | int | ArrayShape,
    dtype: Any = float64,
    order: str = "C",
    name: str | None = None,
) -> Variable:
    if order not in ["C", "F"]:
        raise ValueError(f"Invalid order: {order}, must be 'C' or 'F'")

    fortran_order = order == "F"
    if not isinstance(shape, ArrayShape):
        if not isinstance(shape, (tuple, list)):
            shape = (shape,)
        shape = ArrayShape(tuple(shape), fortran_order=fortran_order)

    dtype = get_datatype(dtype)

    allocate = shape.has_comptime_undefined_dims()
    array = BuilderHelper.generate_local_variables(
        "fc_a",
        name=name,
        dtype=dtype,
        shape=shape,
        allocate=allocate,
    )

    return array
