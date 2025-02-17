from .function import Function
from numeta.syntax.tools import check_node
from numeta.syntax.syntax_settings import settings


class UnaryIntrinsicFunction(Function):
    token = ""

    def __init__(self, argument):
        from numeta.syntax.module import builtins_module

        super().__init__(self.token, [check_node(argument)], module=builtins_module)


class BinaryIntrinsicFunction(Function):
    token = ""

    def __init__(self, argument1, argument2):
        from numeta.syntax.module import builtins_module

        super().__init__(
            self.token,
            [check_node(argument1), check_node(argument2)],
            module=builtins_module,
        )


class NAryIntrinsicFunction(Function):
    token = ""

    def __init__(self, *arguments):
        from numeta.syntax.module import builtins_module

        super().__init__(self.token, [check_node(arg) for arg in arguments], module=builtins_module)


class Abs(UnaryIntrinsicFunction):
    token = "abs"


class Neg(UnaryIntrinsicFunction):
    token = "-"


class Not(UnaryIntrinsicFunction):
    token = ".not."


class Allocated(UnaryIntrinsicFunction):
    token = "allocated"


class Shape(UnaryIntrinsicFunction):
    token = "shape"


class All(UnaryIntrinsicFunction):
    token = "all"


class Real(UnaryIntrinsicFunction):
    token = "real"


class Imag(UnaryIntrinsicFunction):
    token = "aimag"


class Conjugate(UnaryIntrinsicFunction):
    token = "conjg"


class Complex(Function):
    def __init__(self, real, imaginary, kind=settings.DEFAULT_COMPLEX_KIND):
        self.name = "cmplx"
        self.arguments = [check_node(real), check_node(imaginary), check_node(kind)]


class Transpose(UnaryIntrinsicFunction):
    token = "transpose"


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


class Rank(UnaryIntrinsicFunction):
    token = "rank"


class Size(BinaryIntrinsicFunction):
    token = "size"


class Max(NAryIntrinsicFunction):
    token = "max"


class Maxval(UnaryIntrinsicFunction):
    token = "maxval"


class Min(NAryIntrinsicFunction):
    token = "min"


class Minval(UnaryIntrinsicFunction):
    token = "minval"


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


class Trailz(UnaryIntrinsicFunction):
    token = "trailz"


class Sum(UnaryIntrinsicFunction):
    token = "sum"


class Matmul(BinaryIntrinsicFunction):
    token = "matmul"
