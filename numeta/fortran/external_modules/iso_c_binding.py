from numeta.ast import Variable, ExternalNamespace
from numeta.fortran.fortran_type import FortranType
from numeta.array_shape import UNKNOWN


class IsoCBinding(ExternalNamespace):
    def __init__(self):
        super().__init__("iso_c_binding", None)
        self._initialized = False

    def _ensure_initialized(self):
        try:
            initialized = object.__getattribute__(self, "_initialized")
        except AttributeError:
            object.__setattr__(self, "_initialized", False)
            initialized = False
        if initialized:
            return
        object.__setattr__(self, "_initialized", True)

        from numeta.datatype import (
            int32,
            int64,
            size_t,
            float32,
            float64,
            complex64,
            complex128,
            bool8,
            char,
            c_ptr,
        )

        # First create all the basic type variables (without referencing each other)
        self.c_int32 = Variable("c_int32_t", dtype=int32)
        self.c_int64 = Variable("c_int64_t", dtype=int64)
        self.c_size_t = Variable("c_size_t", dtype=size_t)
        self.c_float = Variable("c_float", dtype=float32)
        self.c_double = Variable("c_double", dtype=float64)
        self.c_float_complex = Variable("c_float_complex", dtype=complex64)
        self.c_double_complex = Variable("c_double_complex", dtype=complex128)
        self.c_bool = Variable("c_bool", dtype=bool8)
        self.c_char = Variable("c_char", dtype=char)
        self.c_ptr = Variable("c_ptr", dtype=c_ptr)

        self.add_variable(
            self.c_int32,
            self.c_int64,
            self.c_size_t,
            self.c_float,
            self.c_double,
            self.c_float_complex,
            self.c_double_complex,
            self.c_bool,
            self.c_char,
            self.c_ptr,
        )

        # Now add methods that reference these variables
        # Note: c_f_pointer uses c_ptr which we've already created above
        self.add_method(
            "c_f_pointer",
            [
                Variable("cptr", dtype=c_ptr),
                Variable("fptr", dtype=c_ptr),
            ],
            bind_c=False,
        )

        self.add_method(
            "c_loc",
            [
                Variable("x", dtype=c_ptr),
            ],
            result_=c_ptr,
            bind_c=False,
        )

    def __getattr__(self, name):
        self._ensure_initialized()
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return super().__getattr__(name)


# Create a lazy wrapper that initializes on first access
class _LazyIsoCBinding:
    _instance = None

    def __getattr__(self, name):
        if self._instance is None:
            self._instance = IsoCBinding()
        return getattr(self._instance, name)


iso_c = _LazyIsoCBinding()
