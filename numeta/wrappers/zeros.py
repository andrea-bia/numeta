from numeta.types_hint import float64
from .empty import empty


def zeros(shape, dtype=float64, order="C"):
    array = empty(shape, dtype=dtype, order=order)
    array[:] = 0.0
    return array
