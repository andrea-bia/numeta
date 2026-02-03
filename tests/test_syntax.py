import pytest
import numpy as np

import numeta as nm
from numeta.array_shape import ArrayShape, SCALAR
from numeta.ast import Variable, Assignment, LiteralNode, DerivedType
from numeta.ast.expressions import GetAttr
from numeta.ast import Do, DoWhile, If, ElseIf, Else
from numeta.ast.statements.tools import print_block
from numeta.fortran.fortran_syntax import render_expr_blocks, render_stmt_lines
from numeta.c.c_syntax import (
    render_expr_blocks as render_expr_blocks_c,
    render_stmt_lines as render_stmt_lines_c,
)
from numeta.ast.expressions import (
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
from numeta.ast.statements import VariableDeclaration, Call
from numeta.ast import Subroutine, Module, Scope
from numeta.ast.settings import settings as syntax_settings
from numeta.settings import settings


def render_expr(expr, backend):
    """Return a string representation of an expression."""
    blocks = render_expr_blocks_c(expr) if backend == "c" else render_expr_blocks(expr)
    return print_block(blocks)


def render_blocks(expr, backend):
    return render_expr_blocks_c(expr) if backend == "c" else render_expr_blocks(expr)


def render_stmt(stmt, backend):
    if backend == "c":
        return render_stmt_lines_c(stmt, indent=0)
    return render_stmt_lines(stmt, indent=0)


def expected_for_backend(backend, fortran, c=None):
    if backend == "c":
        return fortran if c is None else c
    return fortran


def assert_render(expr, backend, *, fortran, c=None):
    expected = expected_for_backend(backend, fortran, c)
    assert render_expr(expr, backend) == expected


def assert_render_stmt(stmt, backend, *, fortran, c=None):
    expected = expected_for_backend(backend, fortran, c)
    assert render_stmt(stmt, backend) == expected


def test_simple_assignment_syntax(backend):
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    y = Variable("y", syntax_settings.DEFAULT_INTEGER)
    stmt = Assignment(x, y, add_to_scope=False)
    assert_render_stmt(
        stmt,
        backend,
        fortran=["x=y\n"],
        c=["x = y;\n"],
    )


def test_literal_node(backend):
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    lit = LiteralNode(5)
    assert_render(lit, backend, fortran="5_c_int64_t\n", c="5\n")


def test_binary_operation_node(backend):
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    y = Variable("y", syntax_settings.DEFAULT_INTEGER)
    expr = x + y
    assert_render(expr, backend, fortran="(x+y)\n", c="(x + y)\n")


def test_getattr_node(backend):
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    expr = GetAttr(x, "tag")
    assert_render(expr, backend, fortran="x%tag\n", c="x.tag\n")


def test_getitem_node(backend):
    arr = Variable("a", syntax_settings.DEFAULT_REAL, shape=(10, 10))
    expr = arr[1, 2]
    assert_render(expr, backend, fortran="a(1, 2)\n", c="a[1][2]\n")


def test_unary_neg_node(backend):
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    expr = -x
    assert_render(expr, backend, fortran="-(x)\n")


def test_eq_ne_nodes(backend):
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    y = Variable("y", syntax_settings.DEFAULT_INTEGER)
    assert_render(x == y, backend, fortran="(x.eq.y)\n", c="(x == y)\n")
    assert_render(x != y, backend, fortran="(x.ne.y)\n", c="(x != y)\n")


def test_re_im_nodes(backend):
    z = Variable("z", syntax_settings.DEFAULT_COMPLEX)
    assert_render(z.real, backend, fortran="z%re\n", c="creal(z)\n")
    assert_render(z.imag, backend, fortran="z%im\n", c="cimag(z)\n")


def test_array_constructor(backend):
    i = Variable("i", syntax_settings.DEFAULT_INTEGER)
    arr = Variable("arr", syntax_settings.DEFAULT_INTEGER, shape=(10, 10))
    expr = render_blocks(ArrayConstructor(arr[1, 1], 5, i), backend)
    expected_fortran = [
        "[",
        "arr",
        "(",
        "1",
        ",",
        " ",
        "1",
        ")",
        ", ",
        "5_c_int64_t",
        ", ",
        "i",
        "]",
    ]
    expected_c = ["{", "arr", "[", "1", "]", "[", "1", "]", ", ", "5", ", ", "i", "}"]
    assert expr == expected_for_backend(backend, expected_fortran, expected_c)


def test_complex_function_default(backend):
    settings.set_default_from_datatype(nm.float64, iso_c=True)
    settings.set_default_from_datatype(nm.complex128, iso_c=True)

    a = Variable("a", syntax_settings.DEFAULT_REAL)
    b = Variable("b", syntax_settings.DEFAULT_REAL)
    expr = Complex(a, b)
    assert_render(expr, backend, fortran="cmplx(a, b, c_double_complex)\n")


def test_complex_function(backend):
    settings.set_default_from_datatype(nm.float64, iso_c=True)
    settings.set_default_from_datatype(nm.complex128, iso_c=True)
    settings.set_default_from_datatype(nm.int64, iso_c=True)

    a = Variable("a", syntax_settings.DEFAULT_REAL)
    b = Variable("b", syntax_settings.DEFAULT_REAL)
    expr = Complex(a, b, kind=8)
    assert_render(expr, backend, fortran="cmplx(a, b, 8_c_int64_t)\n", c="cmplx(a, b, 8)\n")


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
def test_intrinsic_functions_with_updated_variables(func, nargs, token, backend):
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    y = Variable("y", syntax_settings.DEFAULT_INTEGER)
    args = [x] if nargs == 1 else [x, y]
    if func is Size:
        args[1] = 1
    expr = func(*args)
    expected_args = ["x"] if nargs == 1 else ["x", "y"]
    if func is Size:
        expected_args[1] = "1" if backend == "c" else "1_c_int64_t"
    expected = f"{token}({', '.join(expected_args)})\n"
    assert_render(expr, backend, fortran=expected, c=expected)


def test_variable_declaration_scalar(backend):
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    dec = VariableDeclaration(x)
    assert_render_stmt(
        dec,
        backend,
        fortran=["integer(c_int64_t) :: x\n"],
        c=["npy_int64 x;\n"],
    )


def test_variable_declaration_array(backend):
    settings.set_default_from_datatype(nm.float64, iso_c=True)
    a = Variable("a", syntax_settings.DEFAULT_REAL, shape=(5,))
    dec = VariableDeclaration(a)
    assert_render_stmt(
        dec,
        backend,
        fortran=["real(c_double), dimension(0:4) :: a\n"],
        c=["npy_float64 a[5];\n"],
    )


def test_variable_declaration_pointer(backend):
    settings.set_default_from_datatype(nm.float64, iso_c=True)
    p = Variable("p", syntax_settings.DEFAULT_REAL, shape=(10, 10), pointer=True)
    dec = VariableDeclaration(p)
    assert_render_stmt(
        dec,
        backend,
        fortran=["real(c_double), pointer, dimension(:,:) :: p\n"],
        c=["npy_float64 *p;\n"],
    )


def test_variable_declaration_allocatable(backend):
    settings.set_default_from_datatype(nm.float64, iso_c=True)
    arr = Variable("arr", syntax_settings.DEFAULT_REAL, shape=(3, 3), allocatable=True)
    dec = VariableDeclaration(arr)
    assert_render_stmt(
        dec,
        backend,
        fortran=["real(c_double), allocatable, dimension(:,:) :: arr\n"],
        c=["npy_float64 *arr;\n"],
    )


def test_variable_declaration_intent(backend):
    settings.set_default_from_datatype(nm.float64, iso_c=True)
    v = Variable("v", syntax_settings.DEFAULT_REAL, intent="in")
    dec = VariableDeclaration(v)
    assert_render_stmt(
        dec,
        backend,
        fortran=["real(c_double), intent(in), value :: v\n"],
        c=["npy_float64 v;\n"],
    )


def test_variable_declaration_bind_c(backend):
    settings.set_default_from_datatype(nm.float64, iso_c=True)
    v = Variable("v", syntax_settings.DEFAULT_REAL, bind_c=True)
    dec = VariableDeclaration(v)
    assert_render_stmt(
        dec,
        backend,
        fortran=["real(c_double), bind(C, name='v') :: v\n"],
        c=["npy_float64 v;\n"],
    )


def test_variable_declaration_assign_scalar(backend):
    settings.set_default_from_datatype(nm.float64, iso_c=True)
    v = Variable("v", syntax_settings.DEFAULT_REAL, assign=5.0)
    dec = VariableDeclaration(v)
    assert_render_stmt(
        dec,
        backend,
        fortran=["real(c_double) :: v; data v / 5.0_c_double /\n"],
        c=["npy_float64 v = 5.0;\n"],
    )


def test_variable_declaration_assign_array(backend):
    settings.set_default_from_datatype(nm.float64, iso_c=True)
    v = Variable("v", syntax_settings.DEFAULT_REAL, shape=(2, 1), assign=np.array([3.0, 5.0]))
    dec = VariableDeclaration(v)
    assert_render_stmt(
        dec,
        backend,
        fortran=[
            "real(c_double), dimension(0:1, 0:0) :: v; data v / 3.0_c_double, 5.0_c_double /\n"
        ],
        c=["npy_float64 v[2][1] = {3.0, 5.0};\n"],
    )


def test_subroutine_print_lines(backend):
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
    expected_c = [
        "void mysub(x, y) {\n",
        "    /* use iso_c_binding, only c_int64_t */\n",
        "    /* implicit none */\n",
        "    npy_int64 x;\n",
        "    npy_int64 y;\n",
        "    y = x;\n",
        "}\n",
    ]
    assert_render_stmt(sub.get_declaration(), backend, fortran=expected, c=expected_c)


def test_module_print_code(backend):
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
    expected_c = [
        "/* module mymod */\n",
        "    /* implicit none */\n",
        "    /* contains */\n",
        "    void mysub(x) {\n",
        "        /* use iso_c_binding, only c_int64_t */\n",
        "        /* implicit none */\n",
        "        npy_int64 x;\n",
        "    }\n",
        "/* end module mymod */\n",
    ]
    assert_render_stmt(mod.get_declaration(), backend, fortran=expected, c=expected_c)


def test_derived_type_declaration(backend):
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
    expected_c = [
        "struct point {\n",
        "    npy_int64 x;\n",
        "    npy_int64 y;\n",
        "    npy_float64 arr[5];\n",
        "};\n",
    ]
    assert_render_stmt(dt.get_declaration(), backend, fortran=expected, c=expected_c)


def test_do_statement(backend):
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    i = Variable("i", syntax_settings.DEFAULT_INTEGER)
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)

    do = Do(i, 0, 3, add_to_scope=False)
    with do:
        Assignment(x, i + 1)

    expected = ["do i = 0_c_int64_t, 3_c_int64_t\n", "    x=(i+1_c_int64_t)\n", "end do\n"]

    expected_c = [
        "for (i = 0; i <= 3; i += 1) {\n",
        "    x = (i + 1);\n",
        "}\n",
    ]
    for l1, l2 in zip(
        render_stmt(do, backend), expected_for_backend(backend, expected, expected_c)
    ):
        assert l1 == l2


def test_if_statement(backend):
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

    expected_c = [
        "for (i = 0; i <= 3; i += 1) {\n",
        "    if ((i < 5)) {\n",
        "        x = (i + 1);\n",
        "    } else if ((i < 10)) {\n",
        "        x = (i + 2);\n",
        "    } else {\n",
        "        x = 0;\n",
        "    }\n",
        "}\n",
    ]
    for l1, l2 in zip(
        render_stmt(wrapper, backend), expected_for_backend(backend, expected, expected_c)
    ):
        assert l1 == l2


def test_do_while_statement(backend):
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    i = Variable("i", syntax_settings.DEFAULT_INTEGER)
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)

    do = DoWhile(i < 5, add_to_scope=False)
    with do:
        Assignment(x, i + 1)

    expected = ["do while ((i.lt.5_c_int64_t))\n", "    x=(i+1_c_int64_t)\n", "end do\n"]

    expected_c = [
        "while ((i < 5)) {\n",
        "    x = (i + 1);\n",
        "}\n",
    ]
    for l1, l2 in zip(
        render_stmt(do, backend), expected_for_backend(backend, expected, expected_c)
    ):
        assert l1 == l2


def test_update_variables_simple_assignment(backend):
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    y = Variable("y", syntax_settings.DEFAULT_INTEGER)
    stmt = Assignment(x, y, add_to_scope=False)

    new_x = Variable("new_x", syntax_settings.DEFAULT_INTEGER)
    new_y = Variable("new_y", syntax_settings.DEFAULT_INTEGER)
    stmt = stmt.get_with_updated_variables([(x, new_x), (y, new_y)])
    assert_render_stmt(
        stmt,
        backend,
        fortran=["new_x=new_y\n"],
        c=["new_x = new_y;\n"],
    )


def test_update_variables_binary_operation_node(backend):
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    y = Variable("y", syntax_settings.DEFAULT_INTEGER)
    expr = x + y

    new_x = Variable("new_x", syntax_settings.DEFAULT_INTEGER)
    new_y = Variable("new_y", syntax_settings.DEFAULT_INTEGER)
    expr = expr.get_with_updated_variables([(x, new_x), (y, new_y)])
    assert_render(expr, backend, fortran="(new_x+new_y)\n", c="(new_x + new_y)\n")


def test_update_variables_simple_add(backend):
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    expr = x + 5

    new_x = Variable("new_x", syntax_settings.DEFAULT_INTEGER)
    expr = expr.get_with_updated_variables([(x, new_x)])
    assert_render(expr, backend, fortran="(new_x+5_c_int64_t)\n", c="(new_x + 5)\n")


def test_update_variables_getattr_node(backend):
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    expr = GetAttr(x, "tag")
    new_x = Variable("new_x", syntax_settings.DEFAULT_INTEGER)
    expr = expr.get_with_updated_variables([(x, new_x)])
    assert_render(expr, backend, fortran="new_x%tag\n", c="new_x.tag\n")


def test_update_variables_getitem_node(backend):
    arr = Variable("a", syntax_settings.DEFAULT_REAL, shape=(10, 10))
    expr = arr[1, 2]
    new_arr = Variable("new_a", syntax_settings.DEFAULT_REAL, shape=(40, 30))
    expr = expr.get_with_updated_variables([(arr, new_arr)])
    assert_render(expr, backend, fortran="new_a(1, 2)\n", c="new_a[1][2]\n")


def test_update_variables_withgetitem_node(backend):
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    expr = Assignment(x, 5, add_to_scope=False)
    new_x = Variable("new_x", syntax_settings.DEFAULT_INTEGER, shape=(10,))
    expr = expr.get_with_updated_variables([(x, new_x[3])])
    assert_render_stmt(
        expr,
        backend,
        fortran=["new_x(3)=5_c_int64_t\n"],
        c=["new_x[3] = 5;\n"],
    )


def test_update_variables_eq_ne_nodes(backend):
    x = Variable("x", syntax_settings.DEFAULT_INTEGER)
    y = Variable("y", syntax_settings.DEFAULT_INTEGER)
    new_x = Variable("new_x", syntax_settings.DEFAULT_INTEGER)
    new_y = Variable("new_y", syntax_settings.DEFAULT_INTEGER)

    expr = x == y
    expr = expr.get_with_updated_variables([(x, new_x), (y, new_y)])
    assert_render(expr, backend, fortran="(new_x.eq.new_y)\n", c="(new_x == new_y)\n")

    expr = x != y
    expr = expr.get_with_updated_variables([(x, new_x), (y, new_y)])
    assert_render(expr, backend, fortran="(new_x.ne.new_y)\n", c="(new_x != new_y)\n")


def test_update_variables_re_im_nodes(backend):
    z = Variable("z", syntax_settings.DEFAULT_COMPLEX)
    new_z = Variable("new_z", syntax_settings.DEFAULT_COMPLEX)

    z_real = z.real
    z_real = z_real.get_with_updated_variables([(z, new_z)])

    z_imag = z.imag
    z_imag = z_imag.get_with_updated_variables([(z, new_z)])

    assert_render(z_real, backend, fortran="new_z%re\n", c="creal(new_z)\n")
    assert_render(z_imag, backend, fortran="new_z%im\n", c="cimag(new_z)\n")


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
def test_intrinsic_functions(func, nargs, token, backend):
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
        expected_args[1] = "1" if backend == "c" else "1_c_int64_t"
    expected = f"{token}({', '.join(expected_args)})\n"
    assert_render(expr, backend, fortran=expected, c=expected)


def test_update_variables_do_statement(backend):
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

    expected_c = [
        "for (new_i = 0; new_i <= 3; new_i += 1) {\n",
        "    new_x = (new_i + 1);\n",
        "}\n",
    ]
    for l1, l2 in zip(
        render_stmt(do, backend), expected_for_backend(backend, expected, expected_c)
    ):
        assert l1 == l2


def test_update_variables_if_statement(backend):
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

    expected_c = [
        "for (new_i = 0; new_i <= 3; new_i += 1) {\n",
        "    if ((new_i < 5)) {\n",
        "        new_x = (new_i + 1);\n",
        "    } else if ((new_i < 10)) {\n",
        "        new_x = (new_i + 2);\n",
        "    } else {\n",
        "        new_x = 0;\n",
        "    }\n",
        "}\n",
    ]
    for line1, line2 in zip(
        render_stmt(wrapper, backend), expected_for_backend(backend, expected, expected_c)
    ):
        assert line1 == line2, f"Expected: {line2}, but got: {line1}"


def test_update_variables_do_while_statement(backend):
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

    expected_c = [
        "while ((new_i < 5)) {\n",
        "    new_x = (new_i + 1);\n",
        "}\n",
    ]
    for l1, l2 in zip(
        render_stmt(do, backend), expected_for_backend(backend, expected, expected_c)
    ):
        assert l1 == l2


def test_call(backend):
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
    expected_c = [
        "void caller() {\n",
        "    /* use iso_c_binding, only c_int64_t */\n",
        "    /* implicit none */\n",
        "    /* interface */\n",
        "        void callee(x, y) {\n",
        "            /* use iso_c_binding, only c_int64_t */\n",
        "            /* implicit none */\n",
        "            npy_int64 x;\n",
        "            npy_int64 y;\n",
        "        }\n",
        "    /* end interface */\n",
        "    npy_int64 x;\n",
        "    npy_int64 y;\n",
        "    callee(x, y);\n",
        "}\n",
    ]
    for l1, l2 in zip(
        render_stmt(caller.get_declaration(), backend),
        expected_for_backend(backend, expected, expected_c),
    ):
        assert l1 == l2


def test_call_external_module(backend):
    settings.set_default_from_datatype(nm.int64, iso_c=True)
    lib = nm.ExternalModule("module", None, hidden=True)
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
    expected_c = [
        "void mysub() {\n",
        "    /* use iso_c_binding, only c_int64_t */\n",
        "    /* implicit none */\n",
        "    /* interface */\n",
        "        void foo(a) {\n",
        "            /* use iso_c_binding, only c_int64_t */\n",
        "            /* implicit none */\n",
        "            npy_int64 a;\n",
        "        }\n",
        "    /* end interface */\n",
        "    npy_int64 x;\n",
        "    foo(x);\n",
        "}\n",
    ]
    for l1, l2 in zip(
        render_stmt(sub.get_declaration(), backend),
        expected_for_backend(backend, expected, expected_c),
    ):
        assert l1 == l2
