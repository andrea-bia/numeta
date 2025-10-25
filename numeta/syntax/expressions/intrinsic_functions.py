from .function import Function
from numeta.syntax.tools import check_node
from numeta.syntax.settings import settings
from numeta.array_shape import ArrayShape, SCALAR, UNKNOWN


class IntrinsicFunction(Function):
    token = ""

    def __init__(self, *arguments):
        from numeta.syntax.module import builtins_module

        super().__init__(self.token, [check_node(arg) for arg in arguments], parent=builtins_module)

    def get_with_updated_variables(self, variables_couples):
        new_args = [arg.get_with_updated_variables(variables_couples) for arg in self.arguments]
        return type(self)(*new_args)

    @property
    def _ftype(self):
        # default behavior for a lot of intrinsic functions
        return self.arguments[0]._ftype

    @property
    def _shape(self):
        # default behavior for a lot of intrinsic functions
        return self.arguments[0]._shape


class UnaryIntrinsicFunction(IntrinsicFunction):

    def __init__(self, argument):
        super().__init__(check_node(argument))


class BinaryIntrinsicFunction(IntrinsicFunction):

    def __init__(self, argument1, argument2):
        super().__init__(
            check_node(argument1),
            check_node(argument2),
        )


class Abs(UnaryIntrinsicFunction):
    token = "abs"


class Neg(UnaryIntrinsicFunction):
    token = "-"


class Not(UnaryIntrinsicFunction):
    token = ".not."


class Allocated(UnaryIntrinsicFunction):
    token = "allocated"

    @property
    def _ftype(self):
        return settings.DEFAULT_LOGICAL

    @property
    def _shape(self):
        return SCALAR


class All(UnaryIntrinsicFunction):
    token = "all"

    @property
    def _ftype(self):
        return settings.DEFAULT_LOGICAL

    @property
    def _shape(self):
        return SCALAR


class Shape(UnaryIntrinsicFunction):
    token = "shape"

    def __init__(self, argument):
        if argument._shape is SCALAR:
            raise ValueError("The shape intrinsic function cannot be applied to a scalar.")
        super().__init__(argument)

    @property
    def _ftype(self):
        return self.arguments[0]._ftype

    @property
    def _shape(self):
        var_shape = self.arguments[0]._shape
        if var_shape is SCALAR or var_shape is UNKNOWN:
            raise ValueError(
                "The shape intrinsic function can only be applied to variables with a defined shape."
            )
        return ArrayShape((len(var_shape.dims),))


class Real(UnaryIntrinsicFunction):
    token = "real"

    @property
    def _ftype(self):
        return settings.DEFAULT_REAL


class Imag(UnaryIntrinsicFunction):
    token = "aimag"

    @property
    def _ftype(self):
        return settings.DEFAULT_REAL


class Conjugate(UnaryIntrinsicFunction):
    token = "conjg"

    @property
    def _ftype(self):
        return settings.DEFAULT_COMPLEX

    @property
    def _shape(self):
        return self.arguments[0]._shape


class Complex(Function):
    def __init__(self, real, imaginary, kind=None):
        self.name = "cmplx"
        if kind is None:
            kind = settings.DEFAULT_COMPLEX.kind
        self.arguments = [check_node(real), check_node(imaginary), check_node(kind)]

    @property
    def _ftype(self):
        # TODO to fix, not consistent with Optional kind
        return settings.DEFAULT_COMPLEX

    @property
    def _shape(self):
        return self.arguments[0]._shape


class Transpose(UnaryIntrinsicFunction):
    token = "transpose"

    @property
    def shape(self):
        if self.arguments[0]._shape is SCALAR:
            raise ValueError("Cannot transpose a scalar.")
        elif self.arguments[0]._shape is UNKNOWN:
            raise ValueError("Cannot transpose a variable with unknown shape.")
        elif len(self.arguments[0]._shape.dims) != 2:
            raise ValueError("Transpose can only be applied to 2-D arrays.")
        return ArrayShape(self.arguments[0]._shape.dims[::-1])


class Exp(UnaryIntrinsicFunction):
    token = "exp"


class Sqrt(UnaryIntrinsicFunction):
    token = "sqrt"


class Floor(UnaryIntrinsicFunction):
    token = "floor"


class Sin(UnaryIntrinsicFunction):
    token = "sin"


class Cos(UnaryIntrinsicFunction):
    token = "cos"


class Tan(UnaryIntrinsicFunction):
    token = "tan"


class Sinh(UnaryIntrinsicFunction):
    token = "sinh"


class Cosh(UnaryIntrinsicFunction):
    token = "cosh"


class Tanh(UnaryIntrinsicFunction):
    token = "tanh"


class ASin(UnaryIntrinsicFunction):
    token = "asin"


class ACos(UnaryIntrinsicFunction):
    token = "acos"


class ATan(UnaryIntrinsicFunction):
    token = "atan"


class ATan2(BinaryIntrinsicFunction):
    token = "atan2"


class Dotproduct(BinaryIntrinsicFunction):
    token = "dot_product"

    @property
    def _ftype(self):
        return self.arguments[0]._ftype

    @property
    def _shape(self):
        return SCALAR


class Rank(UnaryIntrinsicFunction):
    token = "rank"

    @property
    def _ftype(self):
        return settings.DEFAULT_INTEGER

    @property
    def _shape(self):
        return SCALAR


class Size(BinaryIntrinsicFunction):
    token = "size"

    @property
    def _ftype(self):
        return settings.DEFAULT_INTEGER

    @property
    def _shape(self):
        return SCALAR


class Max(IntrinsicFunction):
    token = "max"

    @property
    def _ftype(self):
        return self.arguments[0]._ftype

    @property
    def _shape(self):
        return SCALAR


class Maxval(UnaryIntrinsicFunction):
    token = "maxval"

    @property
    def _ftype(self):
        return self.arguments[0]._ftype

    @property
    def _shape(self):
        return SCALAR


class Min(IntrinsicFunction):
    token = "min"

    @property
    def _ftype(self):
        return self.arguments[0]._ftype

    @property
    def _shape(self):
        return SCALAR


class Minval(UnaryIntrinsicFunction):
    token = "minval"

    @property
    def _ftype(self):
        return self.arguments[0]._ftype

    @property
    def _shape(self):
        return SCALAR


class Iand(BinaryIntrinsicFunction):
    token = "iand"


class Ior(BinaryIntrinsicFunction):
    token = "ior"


class Xor(BinaryIntrinsicFunction):
    token = "xor"


class Ishft(BinaryIntrinsicFunction):
    token = "ishft"


class Ibset(BinaryIntrinsicFunction):
    token = "ibset"


class Ibclr(BinaryIntrinsicFunction):
    token = "ibclr"


class Popcnt(UnaryIntrinsicFunction):
    token = "popcnt"

    @property
    def _ftype(self):
        return settings.DEFAULT_INTEGER

    @property
    def _shape(self):
        return SCALAR


class Trailz(UnaryIntrinsicFunction):
    token = "trailz"

    @property
    def _ftype(self):
        return settings.DEFAULT_INTEGER

    @property
    def _shape(self):
        return SCALAR


class Sum(UnaryIntrinsicFunction):
    token = "sum"

    @property
    def _ftype(self):
        return self.arguments[0]._ftype

    @property
    def _shape(self):
        return SCALAR


class Matmul(BinaryIntrinsicFunction):
    token = "matmul"

    @property
    def _ftype(self):
        return self.arguments[0]._ftype

    @property
    def _shape(self):
        a_shape = self.arguments[0]._shape
        b_shape = self.arguments[1]._shape
        if len(a_shape.dims) == 1:
            return ArrayShape((b_shape.dims[1],))
        if len(b_shape.dims) == 1:
            return ArrayShape((a_shape.dims[0],))
        return ArrayShape((a_shape.dims[0], b_shape.dims[1]))
