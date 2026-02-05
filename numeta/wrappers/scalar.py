import numpy as np
from numeta.builder_helper import BuilderHelper
from numeta.datatype import DataType, get_datatype


def scalar(dtype: DataType | np.generic, value=None, name=None):
    dtype = get_datatype(dtype)

    var = BuilderHelper.generate_local_variables("fc_s", dtype=dtype, name=name)
    if value is not None:
        var[:] = value
    return var
