from numeta.builder_helper import BuilderHelper


def scalar(type_hint, value=None):
    builder = BuilderHelper.get_current_builder()
    var = builder.generate_local_variables("fc_s", ftype=type_hint.dtype.get_fortran())
    if value is not None:
        var[:] = value
    return var
