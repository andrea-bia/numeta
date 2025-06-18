from typing import Protocol, TypeAlias, Union
import numpy as np
from .datatype import (
    StructType,
    int32_dtype,
    int64_dtype,
    size_t_dtype,
    float32_dtype,
    float64_dtype,
    complex64_dtype,
    complex128_dtype,
    bool8_dtype,
    char_dtype,
)


def class_getitem(cls, value):
    """
    This function is used to create a new class with the given shape and order.
    """
    if isinstance(value, str):
        order = value
        shape = cls.flags["shape"]
        rank = None if shape is None else len(shape)
    else:
        order = cls.flags["order"]
        rank = 1 if not isinstance(value, tuple) else len(value)
        shape = value if isinstance(value, tuple) else (value,)

    result = type(
        f"{cls.__name__}[{rank}d]",
        (cls,),
        {"dtype": cls.dtype, "flags": {"order": order, "shape": shape}},
    )
    return result


class NumetaProtocol(Protocol):
    dtype = None
    flags = {"order": None, "shape": None}

    def __class_getitem__(cls, value):
        return class_getitem(cls, value)

    @property
    def shape(self) -> tuple:
        ...

    @property
    def real(self, /) -> Union["NumetaProtocol", np.ndarray, int, float, complex]:
        ...

    @property
    def imag(self, /) -> Union["NumetaProtocol", np.ndarray, int, float, complex]:
        ...

    @property
    def T(self, /) -> Union["NumetaProtocol", np.ndarray, int, float, complex]:
        ...

    def __add__(self, other, /) -> Union["NumetaProtocol", np.ndarray, int, float, complex]:
        ...

    def __radd__(self, other, /) -> Union["NumetaProtocol", np.ndarray, int, float, complex]:
        ...

    def __mul__(self, other, /) -> Union["NumetaProtocol", np.ndarray, int, float, complex]:
        ...

    def __rmul__(self, other, /) -> Union["NumetaProtocol", np.ndarray, int, float, complex]:
        ...

    def __sub__(self, other, /) -> Union["NumetaProtocol", np.ndarray, int, float, complex]:
        ...

    def __truediv__(self, other, /) -> Union["NumetaProtocol", np.ndarray, int, float, complex]:
        ...

    def __rtruediv__(self, other, /) -> Union["NumetaProtocol", np.ndarray, int, float, complex]:
        ...

    def __getitem__(self, key) -> "NumetaProtocol":
        ...

    def __setitem__(self, key, value) -> "NumetaProtocol":
        ...


def get_struct_from_np_dtype(np_dtype):
    fields = []
    for name, (np_dtype, _) in np_dtype.base.fields.items():
        if np_dtype.base.type in np_to_dtype:
            dtype = np_to_dtype[np_dtype.base.type].dtype
        elif np_dtype.base.fields is not None:
            dtype = get_struct_from_np_dtype(np_dtype)
        else:
            raise ValueError(f"Invalid dtype {np_dtype.base.type}, {np_dtype.fields}")

        shape = None if len(np_dtype.shape) == 0 else np_dtype.shape

        fields.append((name, dtype, shape))

    return StructType(fields)


class dtype(NumetaProtocol, Protocol):
    """
    Protocol to create structured data types or to convert numpy data types to numeta data types.
    """

    dtype = None
    flags = {"order": None, "shape": None}

    def __class_getitem__(cls, value):
        if value in np_to_dtype:
            return np_to_dtype[value]
        elif hasattr(value, "fields"):
            dtype = get_struct_from_np_dtype(value)
        elif isinstance(value, dict):
            fields = [(name, hint.dtype, hint.flags["shape"]) for name, hint in value.items()]
            dtype = StructType(fields)
        elif isinstance(value, tuple):

            def get_field(descr):
                name = descr[0]
                hint = descr[1]
                dimension = None
                if len(descr) == 3:
                    dimension = descr[2] if isinstance(descr[2], tuple) else (descr[2],)
                return name, hint.dtype, dimension

            if isinstance(value[0], tuple):
                fields = [get_field(descr) for descr in value]
                dtype = StructType(fields)
            else:
                fields = [get_field(value)]
            dtype = StructType(fields)
        else:
            raise ValueError("Invalid type for dtype")

        result = type(
            f"{cls.__name__}[derived_type.name]",
            (cls,),
            {"__class_getitem__": class_getitem, "shape": None, "dtype": dtype},
        )

        return result


class int32(NumetaProtocol, Protocol):
    dtype = int32_dtype


class int64(NumetaProtocol, Protocol):
    dtype = int64_dtype


class size_t(NumetaProtocol, Protocol):
    dtype = size_t_dtype


class float32(NumetaProtocol, Protocol):
    dtype = float32_dtype


class float64(NumetaProtocol, Protocol):
    dtype = float64_dtype


class complex64(NumetaProtocol, Protocol):
    dtype = complex64_dtype


class complex128(NumetaProtocol, Protocol):
    dtype = complex128_dtype


class bool8(NumetaProtocol, Protocol):
    dtype = bool8_dtype


class char(NumetaProtocol, Protocol):
    dtype = char_dtype


np_to_dtype = {
    np.int32: int32,
    np.int64: int64,
    np.float32: float32,
    np.float64: float64,
    np.complex64: complex64,
    np.complex128: complex128,
    np.bool_: bool8,
    np.str_: char,
}

def get_datatype(dtype):
    if dtype is int:
        return int64_dtype
    elif dtype is float:
        return float64_dtype
    elif dtype is complex:
        return complex128_dtype
    elif dtype is bool:
        return bool8_dtype
    elif dtype.base.type in np_to_dtype:
        return np_to_dtype[dtype.base.type].dtype
    elif hasattr(dtype, "fields"):
        return get_struct_from_np_dtype(dtype)
    else:
        raise ValueError(f"Invalid dtype {dtype}")

from typing import Any
type comptime = Any
