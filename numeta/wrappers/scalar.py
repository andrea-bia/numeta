import numpy as np
from numeta.builder_helper import BuilderHelper
from numeta.datatype import DataType, FortranType, get_datatype


def scalar(dtype: DataType | FortranType | np.generic, value=None, name=None):
    if isinstance(dtype, FortranType):
        ftype = dtype
    else:
        ftype = get_datatype(dtype).get_fortran()

    var = BuilderHelper.generate_local_variables("fc_s", ftype=ftype, name=name)
    if value is not None:
        var[:] = value
    return var
