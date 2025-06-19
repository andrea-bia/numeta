import numpy as np
from .external_modules.iso_c_binding import iso_c
from .syntax import FortranType, DerivedType
from .settings import settings


class DataType:
    _instances = {}

    @classmethod
    def from_np_dtype(cls, dtype):
        return cls._instances[dtype]

    def __init__(self, np_type, fortran_type, fortran_bind_c_type, cnp_type, capi_cast):
        if np_type in DataType._instances:
            raise ValueError(f"DataType {np_type} already exists")
        DataType._instances[np_type] = self
        self.__np_type = np_type
        self.__cnp_type = cnp_type
        self.__fortran_type = fortran_type
        self.__fortran_bind_c_type = fortran_bind_c_type
        self.__capi_cast = capi_cast

    def is_struct(self):
        return False

    def can_be_value(self):
        return True

    def get_numpy(self):
        return self.__np_type

    def get_cnumpy(self):
        return self.__cnp_type

    def get_fortran(self, bind_c=None):
        if bind_c is None:
            return self.__fortran_bind_c_type if settings.use_c_types else self.__fortran_type
        return self.__fortran_bind_c_type if bind_c else self.__fortran_type

    def get_capi_cast(self, obj):
        return self.__capi_cast(obj)


int32 = DataType(
    np.int32,
    FortranType("integer", 4),
    FortranType("integer", iso_c.c_int32),
    "npy_int32",
    lambda x: f"PyLong_AsLongLong({x})",
)

int64 = DataType(
    np.int64,
    FortranType("integer", 8),
    FortranType("integer", iso_c.c_int64),
    "npy_int64",
    lambda x: f"PyLong_AsLongLong({x})",
)

size_t = DataType(
    None,
    None,
    FortranType("integer", iso_c.c_size_t),
    "npy_intp",
    lambda x: f"PyLong_AsLongLong({x})",
)

float32 = DataType(
    np.float32,
    FortranType("real", 4),
    FortranType("real", iso_c.c_float),
    "npy_float32",
    lambda x: f"PyFloat_AsDouble({x})",
)

float64 = DataType(
    np.float64,
    FortranType("real", 8),
    FortranType("real", iso_c.c_double),
    "npy_float64",
    lambda x: f"PyFloat_AsDouble({x})",
)

complex64 = DataType(
    np.complex64,
    FortranType("complex", 4),
    FortranType("complex", iso_c.c_float_complex),
    "npy_complex64",
    lambda x: f"CMPLX(PyComplex_RealAsDouble({x}), PyComplex_ImagAsDouble({x}))",
)

complex128 = DataType(
    np.complex128,
    FortranType("complex", 8),
    FortranType("complex", iso_c.c_double_complex),
    "npy_complex128",
    lambda x: f"CMPLX(PyComplex_RealAsDouble({x}), PyComplex_ImagAsDouble({x}))",
)

bool8 = DataType(
    np.bool_,
    FortranType("logical", 1),
    FortranType("logical", iso_c.c_bool),
    "npy_bool",
    lambda x: f"PyObject_IsTrue({x})",
)

char = DataType(
    np.str_,
    FortranType("character", 1),
    FortranType("character", iso_c.c_char),
    "npy_str",
    lambda x: f"PyUnicode_AsUTF8({x})",
)


class StructType(DataType):
    _instances = {}

    def __new__(cls, members):
        if tuple(members) in StructType._instances:
            return StructType._instances[tuple(members)]
        return super().__new__(cls)

    def __init__(self, members):
        if hasattr(self, "initialized"):
            return
        self.name = f"struct{len(StructType._instances)}"
        self.members = members
        self.__cnp_type = self.name
        self.__fortran = FortranType(
            "type",
            DerivedType(
                self.name,
                [(name, dtype.get_fortran(), dimension) for name, dtype, dimension in members],
            ),
        )
        StructType._instances[tuple(members)] = self
        self.initialized = True

    def get_numpy(self):
        fields = []
        for name, dtype, dimension in self.members:
            fields.append((name, dtype.get_numpy(), dimension))
        return np.dtype(fields)

    def is_struct(self):
        return True

    def can_be_value(self):
        return False

    def c_declaration(self):
        members = []
        for name, dtype, dimension in self.members:
            dec = f"{dtype.get_cnumpy()} {name}"
            if dimension is not None:
                for d in dimension:
                    dec += f"[{d}]"
            members.append(dec)
        members_str = "; ".join(members)
        return f"typedef struct {{ {members_str} ;}} {self.__cnp_type};\n"

    def np(self):
        raise ValueError("StructType has no numpy type")

    def get_fortran(self, bind_c=None):
        return self.__fortran

    def get_cnumpy(self):
        return self.__cnp_type


def get_struct_from_np_dtype(np_dtype):
    fields = []
    for name, (np_d, _) in np_dtype.base.fields.items():
        if np_d.base.type in DataType._instances:
            dtype = DataType._instances[np_d.base.type]
        elif np_d.base.fields is not None:
            dtype = get_struct_from_np_dtype(np_d)
        else:
            raise ValueError(f"Invalid dtype {np_d.base.type}, {np_d.fields}")

        shape = None if len(np_d.shape) == 0 else np_d.shape
        fields.append((name, dtype, shape))

    return StructType(fields)


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
        return int64
    elif dtype is float:
        return float64
    elif dtype is complex:
        return complex128
    elif dtype is bool:
        return bool8
    if isinstance(dtype, np.dtype):
        base = dtype.base.type
    else:
        base = getattr(dtype, "base", dtype).type if hasattr(dtype, "type") else dtype
        if isinstance(base, np.dtype):
            base = base.type
    if base in np_to_dtype:
        return np_to_dtype[base]
    if hasattr(dtype, "fields"):
        return get_struct_from_np_dtype(dtype)
    raise ValueError(f"Invalid dtype {dtype}")
