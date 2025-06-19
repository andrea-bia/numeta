from numeta.builder_helper import BuilderHelper
from numeta.datatype import DataType


def scalar(dtype: DataType, value=None):
    builder = BuilderHelper.get_current_builder()
    var = builder.generate_local_variables("fc_s", ftype=dtype.get_fortran())
    if value is not None:
        var[:] = value
    return var
