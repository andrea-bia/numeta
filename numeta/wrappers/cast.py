from __future__ import annotations
from typing import Any

from numeta.builder_helper import BuilderHelper
from numeta.datatype import DataType, get_datatype
from numeta.fortran.external_modules.iso_c_binding import iso_c
from numeta.ast.variable import Variable


def cast(variable: Any, dtype: Any) -> Variable:
    dtype = get_datatype(dtype)

    builder = BuilderHelper.get_current_builder()
    pointer = builder.generate_local_variables(
        "fc_v",
        dtype=dtype,
        pointer=True,
    )

    variable.target = True
    iso_c.c_f_pointer(iso_c.c_loc(variable), pointer)

    return pointer
