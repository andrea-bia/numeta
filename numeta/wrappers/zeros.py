from numeta.types_hint import float64
from .empty import empty


def zeros(shape, dtype=float64):
    array = empty(shape, dtype=dtype)
    array[:] = 0.0
    return array
