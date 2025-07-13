import numpy as np
from numeta.builder_helper import BuilderHelper
from numeta.datatype import DataType, float64, FortranType
from numeta.array_shape import ArrayShape


def empty(shape, dtype: DataType | FortranType | np.generic = float64, order="C"):
    if order not in ["C", "F"]:
        raise ValueError(f"Invalid order: {order}, must be 'C' or 'F'")

    fortran_order = order == "F"
    if not isinstance(shape, (tuple, list)):
        shape = (shape,)
    shape = ArrayShape(shape, fortran_order=fortran_order)

    if isinstance(dtype, FortranType):
        ftype = dtype
    elif isinstance(dtype, type):
        if issubclass(dtype, DataType):
            ftype = dtype.get_fortran()
        elif issubclass(dtype, np.generic):
            ftype = DataType.from_np_dtype(dtype).get_fortran()
        else:
            raise TypeError(f"Unsupported dtype class: {dtype}")
    else:
        raise TypeError(f"Expected a numpy or numeta dtype got {type(dtype).__name__}")

    builder = BuilderHelper.get_current_builder()
    array = builder.generate_local_variables(
        "fc_a",
        ftype=ftype,
        shape=shape,
        allocate=True,
    )

    return array
