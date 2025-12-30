import pytest
import numpy as np

import numeta as nm
from numeta.array_shape import ArrayShape, SCALAR
from numeta.syntax import Variable, Assignment, LiteralNode, DerivedType
from numeta.syntax.expressions import GetAttr, Function
from numeta.syntax import Do, DoWhile, If, ElseIf, Else
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
    Complex,
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
    ArrayConstructor,
)
from numeta.syntax.statements import VariableDeclaration, Call
from numeta.syntax import Subroutine, Module, Scope
from numeta.syntax.settings import settings as syntax_settings
from numeta.settings import settings


def render(expr):
    """Return a string representation of an expression."""
    return print_block(expr.get_code_blocks())


def test_simple_assignment_syntax():
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    y = Variable("y", syntax_settings.DEFAULT_INTEGER)
    stmt = Assignment(x, y, add_to_scope=False)
    assert stmt.print_lines() == ["x=y\n"]


def test_literal_node():
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    lit = LiteralNode(5)
    assert render(lit) == "5_c_int64_t\n"


def test_binary_operation_node():
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    y = Variable("y", syntax_settings.DEFAULT_INTEGER)
    expr = x + y
    assert render(expr) == "(x+y)\n"


def test_getattr_node():
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    expr = GetAttr(x, "tag")
    assert render(expr) == "x%tag\n"


def test_getitem_node():
    arr = Variable("a", syntax_settings.DEFAULT_REAL, shape=(10, 10))
    expr = arr[1, 2]
    assert render(expr) == "a(1, 2)\n"


def test_unary_neg_node():
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    expr = -x
    assert render(expr) == "-(x)\n"


def test_eq_ne_nodes():
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    y = Variable("y", syntax_settings.DEFAULT_INTEGER)
    assert render(x == y) == "(x.eq.y)\n"
    assert render(x != y) == "(x.ne.y)\n"


def test_re_im_nodes():
    z = Variable("z", syntax_settings.DEFAULT_COMPLEX)
    assert render(z.real) == "z%re\n"
    assert render(z.imag) == "z%im\n"


def test_array_constructor():
    i = Variable("i", syntax_settings.DEFAULT_INTEGER)
    arr = Variable("arr", syntax_settings.DEFAULT_INTEGER, shape=(10, 10))
    expr = ArrayConstructor(arr[1, 1], 5, i).get_code_blocks()
    expected = ["[", "arr", "(", "1", ",", " ", "1", ")", ", ", "5_c_int64_t", ", ", "i", "]"]
    assert expr == expected


def test_complex_function_default():
    settings.set_default_from_datatype(nm.float64, iso_c=True)
    settings.set_default_from_datatype(nm.complex128, iso_c=True)

    a = Variable("a", syntax_settings.DEFAULT_REAL)
    b = Variable("b", syntax_settings.DEFAULT_REAL)
    expr = Complex(a, b)
    assert render(expr) == "cmplx(a, b, c_double_complex)\n"


def test_complex_function():
    settings.set_default_from_datatype(nm.float64, iso_c=True)
    settings.set_default_from_datatype(nm.complex128, iso_c=True)
    settings.set_default_from_datatype(nm.int64, iso_c=True)

    a = Variable("a", syntax_settings.DEFAULT_REAL)
    b = Variable("b", syntax_settings.DEFAULT_REAL)
    expr = Complex(a, b, kind=8)
    assert render(expr) == "cmplx(a, b, 8_c_int64_t)\n"


@pytest.mark.parametrize(
    "func,nargs,token",
    [
        (Abs, 1, "abs"),
        (Neg, 1, "-"),
        (Not, 1, ".not."),
        (Allocated, 1, "allocated"),
        # TODO(Shape, 1, "shape"),
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
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    y = Variable("y", syntax_settings.DEFAULT_INTEGER)
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
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    dec = VariableDeclaration(x)
    assert dec.print_lines() == ["integer(c_int64_t) :: x\n"]


def test_variable_declaration_array():
    settings.set_default_from_datatype(nm.float64, iso_c=True)
    a = Variable("a", syntax_settings.DEFAULT_REAL, shape=(5,))
    dec = VariableDeclaration(a)
    assert dec.print_lines() == ["real(c_double), dimension(0:4) :: a\n"]


def test_variable_declaration_pointer():
    settings.set_default_from_datatype(nm.float64, iso_c=True)
    p = Variable("p", syntax_settings.DEFAULT_REAL, shape=(10, 10), pointer=True)
    dec = VariableDeclaration(p)
    assert dec.print_lines() == ["real(c_double), pointer, dimension(:,:) :: p\n"]


def test_variable_declaration_allocatable():
    settings.set_default_from_datatype(nm.float64, iso_c=True)
    arr = Variable("arr", syntax_settings.DEFAULT_REAL, shape=(3, 3), allocatable=True)
    dec = VariableDeclaration(arr)
    assert dec.print_lines() == ["real(c_double), allocatable, dimension(:,:) :: arr\n"]


def test_variable_declaration_intent():
    settings.set_default_from_datatype(nm.float64, iso_c=True)
    v = Variable("v", syntax_settings.DEFAULT_REAL, intent="in")
    dec = VariableDeclaration(v)
    assert dec.print_lines() == ["real(c_double), intent(in), value :: v\n"]


def test_variable_declaration_bind_c():
    settings.set_default_from_datatype(nm.float64, iso_c=True)
    v = Variable("v", syntax_settings.DEFAULT_REAL, bind_c=True)
    dec = VariableDeclaration(v)
    assert dec.print_lines() == ["real(c_double), bind(C, name='v') :: v\n"]


def test_variable_declaration_assign_scalar():
    settings.set_default_from_datatype(nm.float64, iso_c=True)
    v = Variable("v", syntax_settings.DEFAULT_REAL, assign=5.0)
    dec = VariableDeclaration(v)
    assert dec.print_lines() == ["real(c_double) :: v; data v / 5.0_c_double /\n"]


def test_variable_declaration_assign_array():
    settings.set_default_from_datatype(nm.float64, iso_c=True)
    v = Variable("v", syntax_settings.DEFAULT_REAL, shape=(2, 1), assign=np.array([3.0, 5.0]))
    dec = VariableDeclaration(v)
    assert dec.print_lines() == [
        "real(c_double), dimension(0:1, 0:0) :: v; data v / 3.0_c_double, 5.0_c_double /\n"
    ]


def test_subroutine_print_lines():
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    x = Variable("x", syntax_settings.DEFAULT_INTEGER, intent="in")
    y = Variable("y", syntax_settings.DEFAULT_INTEGER, intent="out")
    sub = Subroutine("mysub")
    sub.add_variable(x, y)
    with sub.scope:
        Assignment(y, x)
    expected = [
        "subroutine mysub(x, y) bind(C, name='mysub')\n",
        "    use iso_c_binding, only: c_int64_t\n",
        "    implicit none\n",
        "    integer(c_int64_t), intent(in), value :: x\n",
        "    integer(c_int64_t), intent(out) :: y\n",
        "    y=x\n",
        "end subroutine mysub\n",
    ]
    assert sub.print_lines() == expected


def test_module_print_code():
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    x = Variable("x", syntax_settings.DEFAULT_INTEGER, intent="in")
    mod = Module("mymod")
    sub = Subroutine("mysub", parent=mod)
    sub.add_variable(x)
    expected = [
        "module mymod\n",
        "    implicit none\n",
        "    contains\n",
        "    subroutine mysub(x) bind(C, name='mysub')\n",
        "        use iso_c_binding, only: c_int64_t\n",
        "        implicit none\n",
        "        integer(c_int64_t), intent(in), value :: x\n",
        "    end subroutine mysub\n",
        "end module mymod\n",
    ]
    assert mod.print_lines() == expected


def test_derived_type_declaration():
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    settings.set_default_from_datatype(nm.float64, iso_c=True)
    dt = DerivedType(
        "point",
        [
            ("x", syntax_settings.DEFAULT_INTEGER, SCALAR),
            ("y", syntax_settings.DEFAULT_INTEGER, SCALAR),
            ("arr", syntax_settings.DEFAULT_REAL, ArrayShape((5,))),
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


def test_do_statement():
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    i = Variable("i", syntax_settings.DEFAULT_INTEGER)
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)

    do = Do(i, 0, 3, add_to_scope=False)
    with do:
        Assignment(x, i + 1)

    expected = ["do i = 0_c_int64_t, 3_c_int64_t\n", "    x=(i+1_c_int64_t)\n", "end do\n"]

    for l1, l2 in zip(do.print_lines(), expected):
        assert l1 == l2


def test_if_statement():
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    i = Variable("i", syntax_settings.DEFAULT_INTEGER)
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)

    wrapper = Do(i, 0, 3, add_to_scope=False)
    with wrapper:
        with If(i < 5):
            Assignment(x, i + 1)
        with ElseIf(i < 10):
            Assignment(x, i + 2)
        with Else():
            Assignment(x, 0)

    expected = [
        "do i = 0_c_int64_t, 3_c_int64_t\n",
        "    if((i.lt.5_c_int64_t))then\n",
        "        x=(i+1_c_int64_t)\n",
        "    elseif((i.lt.10_c_int64_t))then\n",
        "        x=(i+2_c_int64_t)\n",
        "    else\n",
        "        x=0_c_int64_t\n",
        "    end if\n",
        "end do\n",
    ]

    for l1, l2 in zip(wrapper.print_lines(), expected):
        assert l1 == l2


def test_do_while_statement():
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    i = Variable("i", syntax_settings.DEFAULT_INTEGER)
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)

    do = DoWhile(i < 5, add_to_scope=False)
    with do:
        Assignment(x, i + 1)

    expected = ["do while ((i.lt.5_c_int64_t))\n", "    x=(i+1_c_int64_t)\n", "end do\n"]

    for l1, l2 in zip(do.print_lines(), expected):
        assert l1 == l2


def test_update_variables_simple_assignment():
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    y = Variable("y", syntax_settings.DEFAULT_INTEGER)
    stmt = Assignment(x, y, add_to_scope=False)

    new_x = Variable("new_x", syntax_settings.DEFAULT_INTEGER)
    new_y = Variable("new_y", syntax_settings.DEFAULT_INTEGER)
    stmt = stmt.get_with_updated_variables([(x, new_x), (y, new_y)])
    assert stmt.print_lines() == ["new_x=new_y\n"]


def test_update_variables_binary_operation_node():
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    y = Variable("y", syntax_settings.DEFAULT_INTEGER)
    expr = x + y

    new_x = Variable("new_x", syntax_settings.DEFAULT_INTEGER)
    new_y = Variable("new_y", syntax_settings.DEFAULT_INTEGER)
    expr = expr.get_with_updated_variables([(x, new_x), (y, new_y)])
    assert render(expr) == "(new_x+new_y)\n"


def test_update_variables_simple_add():
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    expr = x + 5

    new_x = Variable("new_x", syntax_settings.DEFAULT_INTEGER)
    expr = expr.get_with_updated_variables([(x, new_x)])
    assert render(expr) == "(new_x+5_c_int64_t)\n"


def test_update_variables_getattr_node():
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    expr = GetAttr(x, "tag")
    new_x = Variable("new_x", syntax_settings.DEFAULT_INTEGER)
    expr = expr.get_with_updated_variables([(x, new_x)])
    assert render(expr) == "new_x%tag\n"


def test_update_variables_getitem_node():
    arr = Variable("a", syntax_settings.DEFAULT_REAL, shape=(10, 10))
    expr = arr[1, 2]
    new_arr = Variable("new_a", syntax_settings.DEFAULT_REAL, shape=(40, 30))
    expr = expr.get_with_updated_variables([(arr, new_arr)])
    assert render(expr) == "new_a(1, 2)\n"


def test_update_variables_withgetitem_node():
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    expr = Assignment(x, 5, add_to_scope=False)
    new_x = Variable("new_x", syntax_settings.DEFAULT_INTEGER, shape=(10,))
    expr = expr.get_with_updated_variables([(x, new_x[3])])
    assert render(expr) == "new_x(3)=5_c_int64_t\n"


def test_update_variables_eq_ne_nodes():
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    y = Variable("y", syntax_settings.DEFAULT_INTEGER)
    new_x = Variable("new_x", syntax_settings.DEFAULT_INTEGER)
    new_y = Variable("new_y", syntax_settings.DEFAULT_INTEGER)

    expr = x == y
    expr = expr.get_with_updated_variables([(x, new_x), (y, new_y)])
    assert render(expr) == "(new_x.eq.new_y)\n"

    expr = x != y
    expr = expr.get_with_updated_variables([(x, new_x), (y, new_y)])
    assert render(expr) == "(new_x.ne.new_y)\n"


def test_update_variables_re_im_nodes():
    z = Variable("z", syntax_settings.DEFAULT_COMPLEX)
    new_z = Variable("new_z", syntax_settings.DEFAULT_COMPLEX)

    z_real = z.real
    z_real = z_real.get_with_updated_variables([(z, new_z)])

    z_imag = z.imag
    z_imag = z_imag.get_with_updated_variables([(z, new_z)])

    assert render(z_real) == "new_z%re\n"
    assert render(z_imag) == "new_z%im\n"


@pytest.mark.parametrize(
    "func,nargs,token",
    [
        (Abs, 1, "abs"),
        (Neg, 1, "-"),
        (Not, 1, ".not."),
        (Allocated, 1, "allocated"),
        # TODO(Shape, 1, "shape"),
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
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    y = Variable("y", syntax_settings.DEFAULT_INTEGER)
    args = [x] if nargs == 1 else [x, y]
    if func is Size:
        args[1] = 1
    expr = func(*args)
    new_x = Variable("new_x", syntax_settings.DEFAULT_INTEGER)
    new_y = Variable("new_y", syntax_settings.DEFAULT_INTEGER)
    expr = expr.get_with_updated_variables([(x, new_x), (y, new_y)])

    expected_args = ["new_x"] if nargs == 1 else ["new_x", "new_y"]
    if func is Size:
        expected_args[1] = "1_c_int64_t"
    expected = f"{token}({', '.join(expected_args)})\n"
    assert render(expr) == expected


def test_update_variables_do_statement():
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    i = Variable("i", syntax_settings.DEFAULT_INTEGER)
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)

    new_i = Variable("new_i", syntax_settings.DEFAULT_INTEGER)
    new_x = Variable("new_x", syntax_settings.DEFAULT_INTEGER)

    with Scope():
        do = Do(i, 0, 3, add_to_scope=False)
        with do:
            Assignment(x, i + 1)

        do = do.get_with_updated_variables([(i, new_i), (x, new_x)])

    expected = [
        "do new_i = 0_c_int64_t, 3_c_int64_t\n",
        "    new_x=(new_i+1_c_int64_t)\n",
        "end do\n",
    ]

    for l1, l2 in zip(do.print_lines(), expected):
        assert l1 == l2


def test_update_variables_if_statement():
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    i = Variable("i", syntax_settings.DEFAULT_INTEGER)
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)

    new_i = Variable("new_i", syntax_settings.DEFAULT_INTEGER)
    new_x = Variable("new_x", syntax_settings.DEFAULT_INTEGER)

    with Scope():

        wrapper = Do(i, 0, 3, add_to_scope=False)
        with wrapper:
            with If(i < 5):
                Assignment(x, i + 1)
            with ElseIf(i < 10):
                Assignment(x, i + 2)
            with Else():
                Assignment(x, 0)

        wrapper = wrapper.get_with_updated_variables([(i, new_i), (x, new_x)])

    expected = [
        "do new_i = 0_c_int64_t, 3_c_int64_t\n",
        "    if((new_i.lt.5_c_int64_t))then\n",
        "        new_x=(new_i+1_c_int64_t)\n",
        "    elseif((new_i.lt.10_c_int64_t))then\n",
        "        new_x=(new_i+2_c_int64_t)\n",
        "    else\n",
        "        new_x=0_c_int64_t\n",
        "    end if\n",
        "end do\n",
    ]

    for line1, line2 in zip(wrapper.print_lines(), expected):
        assert line1 == line2, f"Expected: {line2}, but got: {line1}"


def test_update_variables_do_while_statement():
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    i = Variable("i", syntax_settings.DEFAULT_INTEGER)
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)

    new_i = Variable("new_i", syntax_settings.DEFAULT_INTEGER)
    new_x = Variable("new_x", syntax_settings.DEFAULT_INTEGER)

    with Scope():
        do = DoWhile(i < 5, add_to_scope=False)
        with do:
            Assignment(x, i + 1)

        do = do.get_with_updated_variables([(i, new_i), (x, new_x)])

    expected = [
        "do while ((new_i.lt.5_c_int64_t))\n",
        "    new_x=(new_i+1_c_int64_t)\n",
        "end do\n",
    ]

    for l1, l2 in zip(do.print_lines(), expected):
        assert l1 == l2


def test_call():
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    x = Variable("x", syntax_settings.DEFAULT_INTEGER, intent="in")
    y = Variable("y", syntax_settings.DEFAULT_INTEGER, intent="out")
    callee = Subroutine("callee")
    callee.add_variable(x, y)
    with callee.scope:
        Assignment(y, x)
    caller = Subroutine("caller")
    with caller.scope:
        Call(callee, x, y)

    expected = [
        "subroutine caller() bind(C, name='caller')\n",
        "    use iso_c_binding, only: c_int64_t\n",
        "    implicit none\n",
        "    interface\n",
        "        subroutine callee(x, y) bind(C, name='callee')\n",
        "            use iso_c_binding, only: c_int64_t\n",
        "            implicit none\n",
        "            integer(c_int64_t), intent(in), value :: x\n",
        "            integer(c_int64_t), intent(out) :: y\n",
        "        end subroutine callee\n",
        "    end interface\n",
        "    integer(c_int64_t), intent(in), value :: x\n",
        "    integer(c_int64_t), intent(out) :: y\n",
        "    call callee(x, y)\n",
        "end subroutine caller\n",
    ]
    for l1, l2 in zip(caller.print_lines(), expected):
        assert l1 == l2


def test_call_external_module():
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    lib = nm.syntax.module.ExternalModule("module", None, hidden=True)
    lib.add_method("foo", [Variable("a", syntax_settings.DEFAULT_INTEGER)], None)
    foo = lib.foo
    sub = Subroutine("mysub")
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    sub.add_variable(x)
    with sub.scope:
        foo(x)

    expected = [
        "subroutine mysub() bind(C, name='mysub')\n",
        "    use iso_c_binding, only: c_int64_t\n",
        "    implicit none\n",
        "    interface\n",
        "        subroutine foo(a)\n",
        "            use iso_c_binding, only: c_int64_t\n",
        "            implicit none\n",
        "            integer(c_int64_t) :: a\n",
        "        end subroutine foo\n",
        "    end interface\n",
        "    integer(c_int64_t) :: x\n",
        "    call foo(x)\n",
        "end subroutine mysub\n",
    ]
    for l1, l2 in zip(sub.print_lines(), expected):
        assert l1 == l2
