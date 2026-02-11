from __future__ import annotations
from typing import Any

from numeta.ast import PointerAssignment
from numeta.builder_helper import BuilderHelper
from numeta.array_shape import ArrayShape
from numeta.ast.variable import Variable


def reshape(
    variable: Any, shape: tuple[Any, ...] | list[Any] | int | ArrayShape, order: str = "C"
) -> Variable:
    if order not in ["C", "F"]:
        raise ValueError(f"Invalid order: {order}, must be 'C' or 'F'")
    fortran_order = order == "F"

    if not isinstance(shape, ArrayShape):
        if not isinstance(shape, (tuple, list)):
            shape = (shape,)
        shape = ArrayShape(tuple(shape), fortran_order=fortran_order)

    builder = BuilderHelper.get_current_builder()
    pointer = builder.generate_local_variables(
        "fc_v",
        dtype=variable.dtype,
        shape=ArrayShape(tuple([None] * shape.rank), fortran_order=fortran_order),
        pointer=True,
    )

    PointerAssignment(pointer, shape, variable)

    return pointer
