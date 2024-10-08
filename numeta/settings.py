from numeta.syntax import settings as syntax_settings, FortranType
from .external_modules.iso_c_binding import iso_c


class Settings:
    def __init__(
        self,
        c_like=False,
        int_precision=64,
        float_precision=64,
        complex_precision=64,
        bool_precision=8,
        char_precision=8,
        order="C",
    ):
        self.c_like = c_like
        if self.c_like:
            syntax_settings.set_array_lower_bound(0)
            syntax_settings.set_subroutine_bind_c()
            syntax_settings.set_derived_type_bind_c()
            syntax_settings.set_force_value()
        self.use_c_types = self.c_like
        self.set_integer(int_precision)
        self.set_real(float_precision)
        self.set_complex(complex_precision)
        self.set_logical(bool_precision)
        self.set_character(char_precision)
        self.order = order

    def set_array_order(self, order):
        if order == "C":
            self.order = "C"
        elif order == "F":
            self.order = "F"
        else:
            raise ValueError(f"Order {order} not supported")

    def set_integer(self, precision):
        if precision == 32:
            kind = iso_c.c_int32 if self.c_like else 4
        elif precision == 64:
            kind = iso_c.c_int64 if self.c_like else 8
        else:
            raise NotImplementedError(f"Integer precision {precision} bit not supported")
        syntax_settings.set_default_integer_kind(kind)
        self.DEFAULT_INTEGER = FortranType("integer", kind)

    def set_real(self, precision):
        if precision == 32:
            kind = iso_c.c_float if self.c_like else 4
        elif precision == 64:
            kind = iso_c.c_double if self.c_like else 8
        else:
            raise NotImplementedError(f"Real precision {precision} bit not supported")
        syntax_settings.set_default_real_kind(kind)
        self.DEFAULT_REAL = FortranType("real", kind)

    def set_complex(self, precision):
        if precision == 32:
            kind = iso_c.c_float_complex if self.c_like else 4
        elif precision == 64:
            kind = iso_c.c_double_complex if self.c_like else 8
        else:
            raise NotImplementedError(f"Complex precision {precision} bit not supported")
        syntax_settings.set_default_complex_kind(kind)
        self.DEFAULT_COMPLEX = FortranType("complex", kind)

    def set_logical(self, precision):
        if precision == 8:
            kind = iso_c.c_bool if self.c_like else 1
        else:
            raise NotImplementedError(f"Logical precision {precision} bit not supported")
        syntax_settings.set_default_logical_kind(kind)
        self.DEFAULT_LOGICAL = FortranType("logical", kind)

    def set_character(self, precision):
        if precision == 8:
            kind = iso_c.c_char if self.c_like else 1
        else:
            raise NotImplementedError(f"Character precision {precision} bit not supported")
        syntax_settings.set_default_character_kind(kind)
        self.DEFAULT_CHARACTER = FortranType("character", kind)


settings = Settings(c_like=True)
