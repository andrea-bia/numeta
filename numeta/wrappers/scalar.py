from __future__ import annotations
from typing import Any

from numeta.builder_helper import BuilderHelper
from numeta.datatype import DataType, get_datatype
from numeta.ast.variable import Variable


def scalar(dtype: Any, value: Any = None, name: str | None = None) -> Variable:
    dtype = get_datatype(dtype)

    var = BuilderHelper.generate_local_variables("fc_s", dtype=dtype, name=name)
    if value is not None:
        var[:] = value
    return var
