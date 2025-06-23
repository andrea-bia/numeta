import numpy as np
from dataclasses import dataclass
from .external_modules.iso_c_binding import iso_c
from .syntax import FortranType, DerivedType
from .settings import settings


class DataTypeMeta(type):
    """Metaclass used for all data type classes."""

    _instances = {}

    def __new__(mcls, name, bases, attrs):
        cls = super().__new__(mcls, name, bases, attrs)

        # Register built-in numpy mapping when available
        np_type = attrs.get("_np_type")
        if np_type is not None:
            if np_type in DataTypeMeta._instances:
                raise ValueError(f"DataType {np_type} already exists")
            DataTypeMeta._instances[np_type] = cls

        return cls

    def __repr__(cls):
        return f"numeta.{cls._name}"

    def __call__(cls, *args, **kwargs):
        # StructType overrides this behaviour and must be instantiated normally
        value = args[0] if args else kwargs.get("value", None)
        from .wrappers.scalar import scalar

        return scalar(cls, value)


@dataclass(frozen=True)
class ArrayType:
    """Helper object returned by DataType[x] to describe array types."""

    dtype: type
    shape: tuple

    def __iter__(self):
        yield self.dtype
        yield self.shape

    def __repr__(self):
        dims = ",".join(
            ":" if isinstance(d, slice) and d == slice(None) else str(d) for d in self.shape
        )
        return f"{self.dtype.__name__}[{dims}]"

    def __call__(self, *args, **kwargs):
        value = args[0] if args else kwargs.get("value", None)

        from .wrappers.empty import empty

        array = empty(self.shape, dtype=self.dtype, order=kwargs.get("order", "C"))
        array[:] = value

        return array


class DataType(metaclass=DataTypeMeta):
    """Base class for all data type definitions."""

    _np_type = None
    _fortran_type = None
    _fortran_bind_c_type = None
    _cnp_type = None
    _capi_cast = staticmethod(lambda x: x)
    _name = "datatype"
    _is_struct = False
    _can_be_value = True

    @property
    def name(cls):
        return cls._name

    @classmethod
    def __class_getitem__(cls, key) -> ArrayType:
        if not isinstance(key, tuple):
            key = (key,)
        return ArrayType(dtype=cls, shape=tuple(key))

    @classmethod
    def is_np_dtype(cls, dtype):
        return dtype in DataTypeMeta._instances

    @classmethod
    def from_np_dtype(cls, dtype):
        return DataTypeMeta._instances[dtype]

    @classmethod
    def is_struct(cls):
        return cls._is_struct

    @classmethod
    def can_be_value(cls):
        return cls._can_be_value

    @classmethod
    def get_numpy(cls):
        return cls._np_type

    @classmethod
    def get_cnumpy(cls):
        return cls._cnp_type

    @classmethod
    def get_fortran(cls, bind_c=None):
        if bind_c is None:
            return cls._fortran_bind_c_type if settings.use_c_types else cls._fortran_type
        return cls._fortran_bind_c_type if bind_c else cls._fortran_type

    @classmethod
    def get_capi_cast(cls, obj):
        return cls._capi_cast(obj)


class int32(DataType):
    _np_type = np.int32
    _fortran_type = FortranType("integer", 4)
    _fortran_bind_c_type = FortranType("integer", iso_c.c_int32)
    _cnp_type = "npy_int32"
    _capi_cast = staticmethod(lambda x: f"PyLong_AsLongLong({x})")
    _name = "int32"


class int64(DataType):
    _np_type = np.int64
    _fortran_type = FortranType("integer", 8)
    _fortran_bind_c_type = FortranType("integer", iso_c.c_int64)
    _cnp_type = "npy_int64"
    _capi_cast = staticmethod(lambda x: f"PyLong_AsLongLong({x})")
    _name = "int64"


class size_t(DataType):
    _np_type = None
    _fortran_type = None
    _fortran_bind_c_type = FortranType("integer", iso_c.c_size_t)
    _cnp_type = "npy_intp"
    _capi_cast = staticmethod(lambda x: f"PyLong_AsLongLong({x})")
    _name = "size_t"


class float32(DataType):
    _np_type = np.float32
    _fortran_type = FortranType("real", 4)
    _fortran_bind_c_type = FortranType("real", iso_c.c_float)
    _cnp_type = "npy_float32"
    _capi_cast = staticmethod(lambda x: f"PyFloat_AsDouble({x})")
    _name = "float32"


class float64(DataType):
    _np_type = np.float64
    _fortran_type = FortranType("real", 8)
    _fortran_bind_c_type = FortranType("real", iso_c.c_double)
    _cnp_type = "npy_float64"
    _capi_cast = staticmethod(lambda x: f"PyFloat_AsDouble({x})")
    _name = "float64"


class complex64(DataType):
    _np_type = np.complex64
    _fortran_type = FortranType("complex", 4)
    _fortran_bind_c_type = FortranType("complex", iso_c.c_float_complex)
    _cnp_type = "npy_complex64"
    _capi_cast = staticmethod(
        lambda x: f"CMPLX(PyComplex_RealAsDouble({x}), PyComplex_ImagAsDouble({x}))",
    )
    _name = "complex64"


class complex128(DataType):
    _np_type = np.complex128
    _fortran_type = FortranType("complex", 8)
    _fortran_bind_c_type = FortranType("complex", iso_c.c_double_complex)
    _cnp_type = "npy_complex128"
    _capi_cast = staticmethod(
        lambda x: f"CMPLX(PyComplex_RealAsDouble({x}), PyComplex_ImagAsDouble({x}))"
    )
    _name = "complex128"


class bool8(DataType):
    _np_type = np.bool_
    _fortran_type = FortranType("logical", 1)
    _fortran_bind_c_type = FortranType("logical", iso_c.c_bool)
    _cnp_type = "npy_bool"
    _capi_cast = staticmethod(lambda x: f"PyObject_IsTrue({x})")
    _name = "bool8"


class char(DataType):
    _np_type = np.str_
    _fortran_type = FortranType("character", 1)
    _fortran_bind_c_type = FortranType("character", iso_c.c_char)
    _cnp_type = "npy_str"
    _capi_cast = staticmethod(lambda x: f"PyUnicode_AsUTF8({x})")
    _name = "char"


class StructType(DataType, metaclass=DataTypeMeta):
    """Metaclass used to build struct datatype classes."""

    _counter = 0
    _fortran_type = None
    _fortran_bind_c_type = None
    _cnp_type = None
    _capi_cast = staticmethod(lambda x: x)
    _name = "datatype"
    _is_struct = False
    _can_be_value = True
    _members = []

    @property
    def _np_type(cls):
        fields = []
        for mname, dt, dim in cls._members:
            fields.append((mname, dt.get_numpy(), dim))
        return np.dtype(fields)

    @classmethod
    def c_declaration(cls):
        members_str = []
        for mname, dt, dim in cls._members:
            dec = f"{dt.get_cnumpy()} {mname}"
            if dim is not None:
                for d in dim:
                    dec += f"[{d}]"
            members_str.append(dec)
        members_join = "; ".join(members_str)
        return f"typedef struct {{ {members_join} ;}} {cls._cnp_type};\n"

    @classmethod
    def __class_getitem__(cls, key) -> ArrayType:
        if not isinstance(key, tuple):
            key = (key,)
        return ArrayType(dtype=cls, shape=tuple(key))


def make_struct_type(members, name=None):
    """Create (or retrieve) a struct datatype class for ``members``."""

    key = tuple(members)
    if key in DataTypeMeta._instances:
        return DataTypeMeta._instances[key]

    if name is None:
        name = f"struct{StructType._counter}"

    StructType._counter += 1

    fortran_type = FortranType(
        "type",
        DerivedType(
            name,
            [(mname, dt.get_fortran(), dim) for mname, dt, dim in members],
        ),
    )

    attrs = {
        "_name": name,
        "name": name,
        "_members": members,
        "members": members,
        "_cnp_type": name,
        "_fortran_type": fortran_type,
        "_fortran_bind_c_type": fortran_type,
        "_is_struct": True,
        "_can_be_value": False,
    }

    new_cls = type(name, (StructType,), attrs)

    DataTypeMeta._instances[key] = new_cls
    # also register mapping from numpy dtype for conversion
    DataTypeMeta._instances[new_cls.get_numpy()] = new_cls
    return new_cls


def get_struct_from_np_dtype(np_dtype):
    if np_dtype in DataTypeMeta._instances:
        return DataTypeMeta._instances[np_dtype]

    fields = []
    for name, (np_d, _) in np_dtype.base.fields.items():
        if np_d in DataTypeMeta._instances:
            dtype = DataTypeMeta._instances[np_d]
        elif np_d.base.type in DataTypeMeta._instances:
            dtype = DataTypeMeta._instances[np_d.base.type]
        elif np_d.fields is not None or (np_d.base.fields is not None):
            dtype = get_struct_from_np_dtype(np_d)
        else:
            raise ValueError(f"Invalid dtype {np_d.base.type}, {np_d.fields}")

        shape = None if len(np_d.shape) == 0 else np_d.shape
        fields.append((name, dtype, shape))

    struct_cls = make_struct_type(fields)
    DataTypeMeta._instances[np_dtype] = struct_cls
    return struct_cls


def get_datatype(dtype):
    if isinstance(dtype, type) and issubclass(dtype, DataType):
        return dtype

    # Python types
    if dtype is int:
        return int64
    elif dtype is float:
        return float64
    elif dtype is complex:
        return complex128
    elif dtype is bool:
        return bool8

    # Numpy types
    if isinstance(dtype, np.dtype):
        base = dtype.base.type
    else:
        base = getattr(dtype, "base", dtype).type if hasattr(dtype, "type") else dtype
        if isinstance(base, np.dtype):
            base = base.type
    if DataType.is_np_dtype(base):
        return DataType.from_np_dtype(base)

    # Numpy struct
    if hasattr(dtype, "fields"):
        return get_struct_from_np_dtype(dtype)

    raise ValueError(f"Invalid dtype {dtype}")
