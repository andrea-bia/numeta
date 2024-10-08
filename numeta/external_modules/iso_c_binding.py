from numeta.syntax import Variable, ExternalModule, FortranType


class IsoCBinding(ExternalModule):
    def __init__(self):
        super().__init__("iso_c_binding")

        self.c_int32 = Variable("c_int32_t", ftype=None)
        self.c_int64 = Variable("c_int64_t", ftype=None)
        self.c_size_t = Variable("c_size_t", ftype=None)
        self.c_float = Variable("c_float", ftype=None)
        self.c_double = Variable("c_double", ftype=None)
        self.c_float_complex = Variable("c_float_complex", ftype=None)
        self.c_double_complex = Variable("c_double_complex", ftype=None)
        self.c_bool = Variable("c_bool", ftype=None)
        self.c_char = Variable("c_char", ftype=None)

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
        )


iso_c = IsoCBinding()

FInt32_c = FortranType("integer", iso_c.c_int32)
FInt64_c = FortranType("integer", iso_c.c_int64)
FSizet_c = FortranType("integer", iso_c.c_size_t)
FReal32_c = FortranType("real", iso_c.c_float)
FReal64_c = FortranType("real", iso_c.c_double)
FComplex32_c = FortranType("complex", iso_c.c_float_complex)
FComplex64_c = FortranType("complex", iso_c.c_double_complex)
FLogical_c = FortranType("logical", iso_c.c_bool)
FCharacter_c = FortranType("character", iso_c.c_char)
