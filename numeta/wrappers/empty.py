import numpy as np
from numeta.builder_helper import BuilderHelper
from numeta.datatype import DataType, float64, FortranType, get_datatype
from numeta.array_shape import ArrayShape


def empty(shape, dtype: DataType | FortranType | np.generic = float64, order="C", name=None):
    if order not in ["C", "F"]:
        raise ValueError(f"Invalid order: {order}, must be 'C' or 'F'")

    fortran_order = order == "F"
    if not isinstance(shape, ArrayShape):
        if not isinstance(shape, (tuple, list)):
            shape = (shape,)
        shape = ArrayShape(shape, fortran_order=fortran_order)

    if isinstance(dtype, FortranType):
        ftype = dtype
    else:
        ftype = get_datatype(dtype).get_fortran()

    allocate = shape.has_comptime_undefined_dims()
    array = BuilderHelper.generate_local_variables(
        "fc_a",
        name=name,
        ftype=ftype,
        shape=shape,
        allocate=allocate,
    )

    return array
