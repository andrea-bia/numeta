import numpy as np

from numeta.builder_helper import BuilderHelper
from numeta.datatype import DataType, FortranType, get_datatype
from numeta.external_modules.iso_c_binding import iso_c


def cast(variable, dtype: DataType | FortranType | np.generic):

    if isinstance(dtype, FortranType):
        ftype = dtype
    else:
        ftype = get_datatype(dtype).get_fortran()

    builder = BuilderHelper.get_current_builder()
    pointer = builder.generate_local_variables(
        "fc_v",
        ftype=ftype,
        pointer=True,
    )

    variable.target = True
    iso_c.c_f_pointer(iso_c.c_loc(variable), pointer)

    return pointer
