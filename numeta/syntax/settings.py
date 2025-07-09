from .fortran_type import FortranType


class SyntaxSettings:
    def __init__(
        self,
        c_like: bool = False,
        int_kind: int = 8,
        float_kind: int = 8,
        complex_kind: int = 8,
        bool_kind: int = 1,
        char_kind: int = 1,
        order: str = "F",
    ):
        self.__subroutine_bind_c = False
        self.__derived_type_bind_c = False
        self.__array_lower_bound = 1
        self.__c_like_bounds = False
        self.__force_value = False
        self.order = order
        self.c_like = c_like

        if self.c_like:
            self.set_c_like()
        else:
            self.unset_c_like()

        self.set_default_integer_kind(int_kind)
        self.set_default_real_kind(float_kind)
        self.set_default_complex_kind(complex_kind)
        self.set_default_logical_kind(bool_kind)
        self.set_default_character_kind(char_kind)

    def set_c_like(self):
        self.c_like = True
        self.set_array_lower_bound(0)
        self.set_subroutine_bind_c()
        self.set_derived_type_bind_c()
        self.set_force_value()
        self.set_c_like_bounds()
        self.order = "C"

    def unset_c_like(self):
        self.c_like = False
        self.set_array_lower_bound(1)
        self.unset_subroutine_bind_c()
        self.unset_derived_type_bind_c()
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

    # --- Direct FortranType setters --------------------------------------
    def set_default_fortran_type(self, ftype: FortranType):
        if not isinstance(ftype, FortranType):
            raise TypeError("ftype must be a FortranType")
        ftype_name = ftype.type
        if ftype_name == "integer":
            self.set_default_integer_kind(ftype.kind)
        elif ftype_name == "real":
            self.set_default_real_kind(ftype.kind)
        elif ftype_name == "complex":
            self.set_default_complex_kind(ftype.kind)
        elif ftype_name == "logical":
            self.set_default_logical_kind(ftype.kind)
        elif ftype_name == "character":
            self.set_default_character_kind(ftype.kind)
        else:
            raise NotImplementedError(f"Unsupported Fortran type {ftype_name}")

    # --- Kind setters ----------------------------------------------------
    def set_default_integer_kind(self, kind):
        self.DEFAULT_INTEGER_KIND = kind
        self.DEFAULT_INTEGER = FortranType("integer", kind)

    def set_default_real_kind(self, kind):
        self.DEFAULT_REAL_KIND = kind
        self.DEFAULT_REAL = FortranType("real", kind)

    def set_default_complex_kind(self, kind):
        self.DEFAULT_COMPLEX_KIND = kind
        self.DEFAULT_COMPLEX = FortranType("complex", kind)

    def set_default_logical_kind(self, kind):
        self.DEFAULT_LOGICAL_KIND = kind
        self.DEFAULT_LOGICAL = FortranType("logical", kind)

    def set_default_character_kind(self, kind):
        self.DEFAULT_CHARACTER_KIND = kind
        self.DEFAULT_CHARACTER = FortranType("character", kind)

    # --- Properties ------------------------------------------------------
    @property
    def subroutine_bind_c(self):
        return self.__subroutine_bind_c

    def set_subroutine_bind_c(self):
        self.__subroutine_bind_c = True

    def unset_subroutine_bind_c(self):
        self.__subroutine_bind_c = False

    @property
    def derived_type_bind_c(self):
        return self.__derived_type_bind_c

    def set_derived_type_bind_c(self):
        self.__derived_type_bind_c = True

    def unset_derived_type_bind_c(self):
        self.__derived_type_bind_c = False

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
