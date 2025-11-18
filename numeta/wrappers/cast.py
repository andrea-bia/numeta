import numpy as np

from numeta.builder_helper import BuilderHelper
from numeta.datatype import DataType, FortranType
from numeta.external_modules.iso_c_binding import iso_c


def cast(variable, dtype):

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
    pointer = builder.generate_local_variables(
        "fc_v",
        ftype=ftype,
        pointer=True,
    )

    variable.target = True
    iso_c.c_f_pointer(iso_c.c_loc(variable), pointer)

    return pointer
