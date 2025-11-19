import numpy as np
from numeta.builder_helper import BuilderHelper
from numeta.datatype import DataType, FortranType, get_datatype
from numeta.array_shape import ArrayShape


def constant(value, dtype: DataType | FortranType | np.generic | None = None, order="C", name=None):
    if order not in ["C", "F"]:
        raise ValueError(f"Invalid order: {order}, must be 'C' or 'F'")

    builder = BuilderHelper.get_current_builder()

    # Determine value shape and numpy representation
    if isinstance(value, np.ndarray):
        np_value = value
    elif isinstance(value, (list, tuple)):
        np_value = np.array(value)
    else:
        np_value = None

    if np_value is not None:
        shape = np_value.shape
        fortran_order = (
            np_value.flags.f_contiguous if isinstance(value, np.ndarray) else order == "F"
        )
        # Determine datatype if not provided
        if dtype is None:
            dtype = DataType.from_np_dtype(np_value.dtype.type)
        assign_value = np_value
    else:
        shape = None
        fortran_order = True
        if dtype is None:
            np_dtype = np.array(value).dtype.type
            dtype = DataType.from_np_dtype(np_dtype)
        assign_value = value

    # Convert dtype to FortranType
    if isinstance(dtype, FortranType):
        ftype = dtype
    else:
        ftype = get_datatype(dtype).get_fortran()

    if name is None:
        name = "fc_c"

    if shape is None or shape == ():
        if isinstance(assign_value, (int, float, complex)):
            assign_final = str(assign_value)
        else:
            assign_final = assign_value

        return builder.generate_local_variables(
            name,
            ftype=ftype,
            # TODO
            # parameter=True, # parameter is not supported yet, so not really constant.
            # parameter=True,
            assign=assign_final,
        )

    array_shape = ArrayShape(shape, fortran_order=fortran_order)
    return builder.generate_local_variables(
        name,
        ftype=ftype,
        shape=array_shape,
        # TODO
        # parameter=True, # parameter is not supported yet, so not really constant.
        # parameter=True,
        assign=assign_value,
    )
