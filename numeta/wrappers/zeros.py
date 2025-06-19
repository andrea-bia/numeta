from numeta.datatype import DataType, float64
from .empty import empty


def zeros(shape, dtype: DataType = float64, order="C"):
    array = empty(shape, dtype=dtype, order=order)
    array[:] = 0.0
    return array
