import numpy as np
from numeta.builder_helper import BuilderHelper
from numeta.datatype import DataType
from numeta.syntax import FortranType


def scalar(dtype: DataType | FortranType | np.generic, value=None):
    builder = BuilderHelper.get_current_builder()
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

    var = builder.generate_local_variables("fc_s", ftype=ftype)
    if value is not None:
        var[:] = value
    return var
