from numeta.ast.tools import check_node
from numeta.ast.settings import settings
from numeta.array_shape import ArrayShape, SCALAR, UNKNOWN
from .expression_node import ExpressionNode


class IntrinsicFunction(ExpressionNode):
    token = ""

    def __init__(self, *arguments):
        self.arguments = [check_node(arg) for arg in arguments]

    def extract_entities(self):
        for arg in self.arguments:
            yield from arg.extract_entities()

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


class MathIntrinsic(IntrinsicFunction):
    """
    Base class for mathematical intrinsic functions.
    If the argument is integer, it promotes the return type to real (float).
    If the argument is complex, it returns complex.
    If the argument is real, it returns real.
    """

    @property
    def _ftype(self):
        # We look at the first argument to determine the output type base
        arg_type = self.arguments[0]._ftype
        type_name = getattr(arg_type, "type", None)

        if type_name == "integer":
            return settings.DEFAULT_REAL

        # For real or complex, return the same type
        return arg_type


class UnaryMathIntrinsic(MathIntrinsic, UnaryIntrinsicFunction):
    pass


class Abs(UnaryIntrinsicFunction):
    token = "abs"

    @property
    def _ftype(self):
        arg_type = self.arguments[0]._ftype
        type_name = getattr(arg_type, "type", None)

        if type_name == "complex":
            # abs(complex) returns real.
            # Match precision: complex(4) -> real(4), complex(8) -> real(8)
            from numeta.datatype import float32, float64

            kind = getattr(arg_type, "kind", None)
            if str(kind) == "4":
                return float32.get_fortran()
            if str(kind) == "8":
                return float64.get_fortran()

            return settings.DEFAULT_REAL

        return arg_type


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


class Complex(IntrinsicFunction):
    token = "cmplx"

    def __init__(self, real, imaginary, kind=None):
        if kind is None:
            kind = settings.DEFAULT_COMPLEX.kind
        super().__init__(real, imaginary, kind)

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
    def _shape(self):
        arg_shape = self.arguments[0]._shape
        if arg_shape is SCALAR:
            raise ValueError("Cannot transpose a scalar.")
        if arg_shape is UNKNOWN:
            raise ValueError("Cannot transpose a variable with unknown shape.")
        if len(arg_shape.dims) != 2:
            raise ValueError("Transpose can only be applied to 2-D arrays.")
        return ArrayShape(arg_shape.dims[::-1], fortran_order=arg_shape.fortran_order)

    @property
    def shape(self):
        if self.arguments[0]._shape is SCALAR:
            raise ValueError("Cannot transpose a scalar.")
        elif self.arguments[0]._shape is UNKNOWN:
            raise ValueError("Cannot transpose a variable with unknown shape.")
        elif len(self.arguments[0]._shape.dims) != 2:
            raise ValueError("Transpose can only be applied to 2-D arrays.")
        return ArrayShape(self.arguments[0]._shape.dims[::-1])


class Exp(UnaryMathIntrinsic):
    token = "exp"


class Log(UnaryMathIntrinsic):
    token = "log"


class Log10(UnaryMathIntrinsic):
    token = "log10"


class Sqrt(UnaryMathIntrinsic):
    token = "sqrt"


class Floor(UnaryMathIntrinsic):
    token = "floor"


class Ceil(UnaryMathIntrinsic):
    token = "ceil"


class Sin(UnaryMathIntrinsic):
    token = "sin"


class Cos(UnaryMathIntrinsic):
    token = "cos"


class Tan(UnaryMathIntrinsic):
    token = "tan"


class Sinh(UnaryMathIntrinsic):
    token = "sinh"


class Cosh(UnaryMathIntrinsic):
    token = "cosh"


class Tanh(UnaryMathIntrinsic):
    token = "tanh"


class Arcsin(UnaryMathIntrinsic):
    token = "asin"


class Arccos(UnaryMathIntrinsic):
    token = "acos"


class Arctan(UnaryMathIntrinsic):
    token = "atan"


class Arctan2(BinaryIntrinsicFunction, MathIntrinsic):
    token = "atan2"


class Arcsinh(UnaryMathIntrinsic):
    token = "asinh"


class Arccosh(UnaryMathIntrinsic):
    token = "acosh"


class Arctanh(UnaryMathIntrinsic):
    token = "atanh"


class Hypot(BinaryIntrinsicFunction, MathIntrinsic):
    token = "hypot"


class Copysign(BinaryIntrinsicFunction, MathIntrinsic):
    token = "copysign"


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


# Aliases to match numpy conventions
abs = Abs
negative = Neg
logical_not = Not
allocated = Allocated
all = All
shape = Shape
real = Real
imag = Imag
conjugate = Conjugate
conj = Conjugate
complex = Complex
transpose = Transpose
exp = Exp
log = Log
log10 = Log10
sqrt = Sqrt
floor = Floor
ceil = Ceil
sin = Sin
cos = Cos
tan = Tan
sinh = Sinh
cosh = Cosh
tanh = Tanh
arcsin = Arcsin
arccos = Arccos
arctan = Arctan
arctan2 = Arctan2
arcsinh = Arcsinh
arccosh = Arccosh
arctanh = Arctanh
hypot = Hypot
copysign = Copysign
dot = Dotproduct
ndim = Rank
size = Size
maximum = Max
max = Maxval
minimum = Min
min = Minval
bitwise_and = Iand
bitwise_or = Ior
bitwise_xor = Xor
ishft = Ishft
ibset = Ibset
ibclr = Ibclr
popcnt = Popcnt
trailz = Trailz
sum = Sum
matmul = Matmul
