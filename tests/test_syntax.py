import numeta as nm
from numeta.syntax import Variable, Assignment, LiteralNode, DerivedType
from numeta.syntax.expressions import GetAttr, Function
import pytest
from numeta.syntax import sin, sqrt, Complex
from numeta.syntax.statements.tools import print_block
from numeta.syntax.expressions import (
    Abs,
    Neg,
    Not,
    Allocated,
    Shape,
    All,
    Real,
    Imag,
    Conjugate,
    Transpose,
    Exp,
    Sqrt,
    Floor,
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,
    ASin,
    ACos,
    ATan,
    ATan2,
    Dotproduct,
    Rank,
    Size,
    Max,
    Maxval,
    Min,
    Minval,
    Iand,
    Ior,
    Xor,
    Ishft,
    Ibset,
    Ibclr,
    Popcnt,
    Trailz,
    Sum,
    Matmul,
)
from numeta.syntax.statements import VariableDeclaration
from numeta.syntax import Subroutine, Module
from numeta import settings


def render(expr):
    """Return a string representation of an expression."""
    return print_block(expr.get_code_blocks())


def test_simple_assignment_syntax():
    x = Variable("x", nm.settings.DEFAULT_INTEGER)
    y = Variable("y", nm.settings.DEFAULT_INTEGER)
    stmt = Assignment(x, y, add_to_scope=False)
    assert stmt.print_lines() == ["x=y\n"]


def test_literal_node():
    nm.settings.set_integer(64)
    lit = LiteralNode(5)
    assert render(lit) == "5_c_int64_t\n"


def test_binary_operation_node():
    x = Variable("x", nm.settings.DEFAULT_INTEGER)
    y = Variable("y", nm.settings.DEFAULT_INTEGER)
    expr = x + y
    assert render(expr) == "(x+y)\n"


def test_getattr_node():
    x = Variable("x", nm.settings.DEFAULT_INTEGER)
    expr = GetAttr(x, "tag")
    assert render(expr) == "x%tag\n"


def test_getitem_node():
    arr = Variable("a", nm.settings.DEFAULT_REAL, dimension=(10, 10))
    expr = arr[1, 2]
    assert render(expr) == "a(1, 2)\n"


def test_function_node():
    x = Variable("x", nm.settings.DEFAULT_INTEGER)
    fn = Function("f", [x])
    assert render(fn) == "f(x)\n"


def test_unary_neg_node():
    x = Variable("x", nm.settings.DEFAULT_INTEGER)
    expr = -x
    assert render(expr) == "-(x)\n"


def test_eq_ne_nodes():
    x = Variable("x", nm.settings.DEFAULT_INTEGER)
    y = Variable("y", nm.settings.DEFAULT_INTEGER)
    assert render(x == y) == "(x.eq.y)\n"
    assert render(x != y) == "(x.ne.y)\n"


def test_re_im_nodes():
    z = Variable("z", nm.c8)
    assert render(z.real) == "z%re\n"
    assert render(z.imag) == "z%im\n"


def test_function_multiple_args():
    x = Variable("x", nm.settings.DEFAULT_INTEGER)
    y = Variable("y", nm.settings.DEFAULT_INTEGER)
    fn = Function("f", [x, y])
    assert render(fn) == "f(x, y)\n"


def test_intrinsic_function_sin():
    x = Variable("x", nm.f8)
    expr = sin(x)
    assert render(expr) == "sin(x)\n"


def test_intrinsic_function_sqrt():
    x = Variable("x", nm.f8)
    expr = sqrt(x)
    assert render(expr) == "sqrt(x)\n"


def test_complex_function():

    a = Variable("a", nm.f8)
    b = Variable("b", nm.f8)
    expr = Complex(a, b)
    assert render(expr) == "cmplx(a, b, 8_c_int64_t)\n"


@pytest.mark.parametrize(
    "func,nargs,token",
    [
        (Abs, 1, "abs"),
        (Neg, 1, "-"),
        (Not, 1, ".not."),
        (Allocated, 1, "allocated"),
        (Shape, 1, "shape"),
        (All, 1, "all"),
        (Real, 1, "real"),
        (Imag, 1, "aimag"),
        (Conjugate, 1, "conjg"),
        (Transpose, 1, "transpose"),
        (Exp, 1, "exp"),
        (Sqrt, 1, "sqrt"),
        (Floor, 1, "floor"),
        (Sin, 1, "sin"),
        (Cos, 1, "cos"),
        (Tan, 1, "tan"),
        (Sinh, 1, "sinh"),
        (Cosh, 1, "cosh"),
        (Tanh, 1, "tanh"),
        (ASin, 1, "asin"),
        (ACos, 1, "acos"),
        (ATan, 1, "atan"),
        (Rank, 1, "rank"),
        (Maxval, 1, "maxval"),
        (Minval, 1, "minval"),
        (Popcnt, 1, "popcnt"),
        (Trailz, 1, "trailz"),
        (Sum, 1, "sum"),
        (ATan2, 2, "atan2"),
        (Dotproduct, 2, "dot_product"),
        (Size, 2, "size"),
        (Max, 2, "max"),
        (Min, 2, "min"),
        (Iand, 2, "iand"),
        (Ior, 2, "ior"),
        (Xor, 2, "xor"),
        (Ishft, 2, "ishft"),
        (Ibset, 2, "ibset"),
        (Ibclr, 2, "ibclr"),
        (Matmul, 2, "matmul"),
    ],
)
def test_intrinsic_functions(func, nargs, token):
    nm.settings.set_integer(64)
    x = Variable("x", nm.settings.DEFAULT_INTEGER)
    y = Variable("y", nm.settings.DEFAULT_INTEGER)
    args = [x] if nargs == 1 else [x, y]
    if func is Size:
        args[1] = 1
    expr = func(*args)
    expected_args = ["x"] if nargs == 1 else ["x", "y"]
    if func is Size:
        expected_args[1] = "1_c_int64_t"
    expected = f"{token}({', '.join(expected_args)})\n"
    assert render(expr) == expected


def test_variable_declaration_scalar():
    x = Variable("x", settings.DEFAULT_INTEGER)
    dec = VariableDeclaration(x)
    assert dec.print_lines() == ["integer(c_int64_t) :: x\n"]


def test_variable_declaration_array():
    a = Variable("a", settings.DEFAULT_REAL, dimension=(5,))
    dec = VariableDeclaration(a)
    assert dec.print_lines() == ["real(c_double), dimension(0:4) :: a\n"]


def test_variable_declaration_pointer():
    p = Variable("p", settings.DEFAULT_REAL, dimension=(10, 10), pointer=True)
    dec = VariableDeclaration(p)
    assert dec.print_lines() == ["real(c_double), pointer, dimension(:,:) :: p\n"]


def test_variable_declaration_allocatable():
    arr = Variable("arr", settings.DEFAULT_REAL, dimension=(3, 3), allocatable=True)
    dec = VariableDeclaration(arr)
    assert dec.print_lines() == ["real(c_double), allocatable, dimension(:,:) :: arr\n"]


def test_variable_declaration_intent():
    v = Variable("v", settings.DEFAULT_REAL, intent="in")
    dec = VariableDeclaration(v)
    assert dec.print_lines() == ["real(c_double), intent(in), value :: v\n"]


def test_subroutine_print_lines():
    sub = Subroutine("mysub")
    nm.settings.set_integer(64)
    x = Variable("x", settings.DEFAULT_INTEGER, intent="in")
    y = Variable("y", settings.DEFAULT_INTEGER, intent="out")
    sub.add_variable(x, y)
    with sub.scope:
        Assignment(y, x)
    expected = [
        "subroutine mysub(x, y) bind(C)\n",
        "    use iso_c_binding, only: c_int64_t\n",
        "    implicit none\n",
        "    integer(c_int64_t), intent(in), value :: x\n",
        "    integer(c_int64_t), intent(out) :: y\n",
        "    y=x\n",
        "end subroutine mysub\n",
    ]
    assert sub.print_lines() == expected


def test_module_print_code():
    mod = Module("mymod")
    sub = Subroutine("mysub", module=mod)
    nm.settings.set_integer(64)
    x = Variable("x", settings.DEFAULT_INTEGER, intent="in")
    sub.add_variable(x)
    expected = "module mymod\n" "    implicit none\n" "    contains\n" "    subroutine mysub(x) bind(C)\n" "        use iso_c_binding, only: c_int64_t\n" "        implicit none\n" "        integer(c_int64_t), intent(in), value :: x\n" "    end subroutine mysub\n" "end module mymod\n"
    assert mod.get_code() == expected


def test_derived_type_declaration():
    dt = DerivedType(
        "point",
        [
            ("x", settings.DEFAULT_INTEGER, None),
            ("y", settings.DEFAULT_INTEGER, None),
            ("arr", settings.DEFAULT_REAL, (5,)),
        ],
    )
    expected = [
        "type, bind(C) :: point\n",
        "    integer(c_int64_t) :: x\n",
        "    integer(c_int64_t) :: y\n",
        "    real(c_double), dimension(0:4) :: arr\n",
        "end type point\n",
    ]
    assert dt.get_declaration().print_lines() == expected
