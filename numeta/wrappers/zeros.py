import numpy as np
from numeta.datatype import DataType, float64
from numeta.syntax import FortranType
from .empty import empty


def zeros(shape, dtype: DataType | FortranType | np.generic = float64, order="C"):
    array = empty(shape, dtype=dtype, order=order)
    array[:] = 0.0
    return array
