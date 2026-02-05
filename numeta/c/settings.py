class SyntaxSettings:
    def __init__(
        self,
        c_like: bool = True,
        default_int=None,
        default_float=None,
        default_complex=None,
        default_bool=None,
        default_char=None,
        order: str = "C",
    ):
        self.__procedure_bind_c = False
        self.__struct_type_bind_c = False
        self.__array_lower_bound = 0
        self.__c_like_bounds = False
        self.__force_value = False
        self.order = order
        self.c_like = c_like

        if self.c_like:
            self.set_c_like()
        else:
            self.unset_c_like()

        self.DEFAULT_INT = None
        self.DEFAULT_INTEGER = None
        self.DEFAULT_FLOAT = None
        self.DEFAULT_REAL = None
        self.DEFAULT_COMPLEX = None
        self.DEFAULT_BOOL = None
        self.DEFAULT_LOGICAL = None
        self.DEFAULT_CHAR = None
        self.DEFAULT_CHARACTER = None
        if default_int is not None:
            self.set_default_datatype(default_int)
        if default_float is not None:
            self.set_default_datatype(default_float)
        if default_complex is not None:
            self.set_default_datatype(default_complex)
        if default_bool is not None:
            self.set_default_datatype(default_bool)
        if default_char is not None:
            self.set_default_datatype(default_char)

    def set_c_like(self):
        self.c_like = True
        self.set_array_lower_bound(0)
        self.set_procedure_bind_c()
        self.set_struct_type_bind_c()
        self.set_force_value()
        self.set_c_like_bounds()
        self.order = "C"

    def unset_c_like(self):
        self.c_like = False
        self.set_array_lower_bound(1)
        self.unset_procedure_bind_c()
        self.unset_struct_type_bind_c()
        self.unset_force_value()
        self.unset_c_like_bounds()
        self.order = "F"

    def set_array_order(self, order: str):
        if order == "C":
            self.order = "C"
        elif order == "F":
            self.order = "F"
        else:
            raise ValueError(f"Order {order} not supported")

    # --- Direct DataType setters --------------------------------------
    def set_default_datatype(self, dtype):
        from numeta.datatype import (
            DataType,
            int32,
            int64,
            float32,
            float64,
            complex64,
            complex128,
            bool8,
            char,
        )

        if not (isinstance(dtype, type) and issubclass(dtype, DataType)):
            raise TypeError("dtype must be a DataType subclass")

        if dtype in (int32, int64):
            self.DEFAULT_INT = dtype
            self.DEFAULT_INTEGER = self.DEFAULT_INT
        elif dtype in (float32, float64):
            self.DEFAULT_FLOAT = dtype
            self.DEFAULT_REAL = self.DEFAULT_FLOAT
        elif dtype in (complex64, complex128):
            self.DEFAULT_COMPLEX = dtype
        elif dtype == bool8:
            self.DEFAULT_BOOL = dtype
            self.DEFAULT_LOGICAL = self.DEFAULT_BOOL
        elif dtype == char:
            self.DEFAULT_CHAR = dtype
            self.DEFAULT_CHARACTER = self.DEFAULT_CHAR
        else:
            raise NotImplementedError(f"Unsupported DataType {dtype}")

    # --- Properties ------------------------------------------------------
    @property
    def procedure_bind_c(self):
        return self.__procedure_bind_c

    def set_procedure_bind_c(self):
        self.__procedure_bind_c = True

    def unset_procedure_bind_c(self):
        self.__procedure_bind_c = False

    @property
    def struct_type_bind_c(self):
        return self.__struct_type_bind_c

    def set_struct_type_bind_c(self):
        self.__struct_type_bind_c = True

    def unset_struct_type_bind_c(self):
        self.__struct_type_bind_c = False

    @property
    def array_lower_bound(self):
        return self.__array_lower_bound

    def set_array_lower_bound(self, value):
        try:
            self.__array_lower_bound = int(value)
        except ValueError as e:
            raise ValueError(f"Array lower bound must be an integer, got {value}") from e

    @property
    def c_like_bounds(self):
        return self.__c_like_bounds

    def set_c_like_bounds(self):
        self.__c_like_bounds = True

    def unset_c_like_bounds(self):
        self.__c_like_bounds = False

    @property
    def force_value(self):
        return self.__force_value

    def set_force_value(self):
        self.__force_value = True

    def unset_force_value(self):
        self.__force_value = False


settings = SyntaxSettings()
