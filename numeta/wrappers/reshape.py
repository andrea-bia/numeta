from __future__ import annotations
from typing import Any

from numeta.ast import PointerAssignment
from numeta.builder_helper import BuilderHelper
from numeta.array_shape import ArrayShape, SCALAR, UNKNOWN
from numeta.datatype import size_t
from numeta.ast.variable import Variable


def reshape(
    variable: Any,
    shape: tuple[Any, ...] | list[Any] | int | ArrayShape | Variable,
    order: str = "C",
) -> Variable:
    def normalize_shape_argument(shape_arg):
        if isinstance(shape_arg, ArrayShape):
            return shape_arg

        if isinstance(shape_arg, (tuple, list)):
            return tuple(shape_arg)

        shape_meta = getattr(shape_arg, "_shape", None)
        if isinstance(shape_meta, ArrayShape) and shape_meta not in (SCALAR, UNKNOWN):
            if shape_meta.rank != 1:
                raise ValueError("Shape array argument must be rank-1.")
            if shape_meta.rank == 0 or not isinstance(shape_meta.dim(0), int):
                raise ValueError("Shape array argument must have compile-time known length.")

            rank = shape_meta.dim(0)
            if isinstance(shape_arg, Variable):
                return ArrayShape.from_shape_vector(shape_arg, rank, fortran_order=fortran_order)

            tmp_shape = BuilderHelper.generate_local_variables(
                "nm_shape", dtype=size_t, shape=ArrayShape((rank,))
            )
            tmp_shape[:] = shape_arg
            return ArrayShape.from_shape_vector(tmp_shape, rank, fortran_order=fortran_order)

        return (shape_arg,)

    if order not in ["C", "F"]:
        raise ValueError(f"Invalid order: {order}, must be 'C' or 'F'")
    fortran_order = order == "F"

    normalized_shape_arg = normalize_shape_argument(shape)
    if isinstance(normalized_shape_arg, ArrayShape):
        shape = normalized_shape_arg
    else:
        shape = ArrayShape(tuple(normalized_shape_arg), fortran_order=fortran_order)

    shape = BuilderHelper.normalize_allocation_shape(shape)

    builder = BuilderHelper.get_current_builder()
    pointer = builder.generate_local_variables(
        "fc_v",
        dtype=variable.dtype,
        shape=shape,
        pointer=True,
    )

    PointerAssignment(pointer, shape, variable)

    return pointer
