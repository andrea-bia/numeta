from __future__ import annotations

from typing import Any, cast

import numpy as np

from numeta.array_shape import ArrayShape, SCALAR, UNKNOWN
from numeta.datatype import DataType
from numeta.ast.statements.tools import print_block
from numeta.c.settings import settings as syntax_settings

from numeta.fortran.fortran_type import FortranType
from numeta.ast.nodes import NamedEntity
from numeta.ast.variable import Variable
from numeta.ast.expressions import (
    BinaryOperationNode,
    FunctionCall,
    GetAttr,
    GetItem,
    IntrinsicFunction,
    LiteralNode,
)
from numeta.ast.expressions.binary_operation_node import BinaryOperationNodeNoPar
from numeta.ast.expressions.various import ArrayConstructor, Im, Re
from numeta.ast.statements import (
    Allocate,
    Assignment,
    Call,
    Case,
    Contains,
    Deallocate,
    Do,
    DoWhile,
    Else,
    ElseIf,
    If,
    Implicit,
    Interface,
    Return,
    SelectCase,
    Use,
)
from numeta.ast.statements.derived_type_declaration import DerivedTypeDeclaration
from numeta.ast.statements.function_declaration import FunctionInterfaceDeclaration
from numeta.ast.statements.module_declaration import ModuleDeclaration
from numeta.ast.statements.subroutine_declaration import InterfaceDeclaration, SubroutineDeclaration
from numeta.ast.statements.variable_declaration import VariableDeclaration
from numeta.ast.statements.various import Comment, PointerAssignment, Print, SimpleStatement


_C_BINARY_OPS = {
    ".eq.": "==",
    ".ne.": "!=",
    ".lt.": "<",
    ".le.": "<=",
    ".gt.": ">",
    ".ge.": ">=",
    ".and.": "&&",
    ".or.": "||",
}


def _literal_blocks_from_ftype(value: Any, ftype: Any) -> list[str]:
    ftype_type = getattr(ftype, "type", None)
    if ftype_type == "type":
        return [f"{value}"]
    if ftype_type == "integer":
        return [str(int(value))]
    if ftype_type == "real":
        return [str(float(value))]
    if ftype_type == "complex":
        return ["(", str(value.real), " + ", str(value.imag), "*I", ")"]
    if ftype_type == "logical":
        return ["1" if value is True else "0"]
    if ftype_type == "character":
        return [f'"{value}"']
    raise ValueError(f"Unknown type: {ftype_type}")


def render_expr_blocks(
    expr: Any,
    *,
    shape_arg_map: dict[str, str] | None = None,
) -> list[str]:
    if expr is None:
        return [""]
    if isinstance(expr, LiteralNode):
        return _literal_blocks_from_ftype(expr.value, cast(FortranType, expr._ftype))
    if isinstance(expr, (int, float, complex, bool, str, np.generic)):
        literal = LiteralNode(expr)
        return _literal_blocks_from_ftype(literal.value, cast(FortranType, literal._ftype))
    if isinstance(expr, Variable):
        if shape_arg_map is not None and expr.name in shape_arg_map:
            return [f"{shape_arg_map[expr.name]}_dims"]
        return [expr.name]
    if isinstance(expr, NamedEntity):
        if shape_arg_map is not None and expr.name in shape_arg_map:
            return [f"{shape_arg_map[expr.name]}_dims"]
        return [expr.name]
    if isinstance(expr, BinaryOperationNodeNoPar):
        op_value = _C_BINARY_OPS.get(expr.op, expr.op)
        op = op_value if op_value is not None else ""
        if expr.op == "**":
            return [
                "pow",
                "(",
                *render_expr_blocks(expr.left, shape_arg_map=shape_arg_map),
                ", ",
                *render_expr_blocks(expr.right, shape_arg_map=shape_arg_map),
                ")",
            ]
        return [
            *render_expr_blocks(expr.left, shape_arg_map=shape_arg_map),
            " ",
            op,
            " ",
            *render_expr_blocks(expr.right, shape_arg_map=shape_arg_map),
        ]
    if isinstance(expr, BinaryOperationNode):
        op_value = _C_BINARY_OPS.get(expr.op, expr.op)
        op = op_value if op_value is not None else ""
        if expr.op == "**":
            return [
                "pow",
                "(",
                *render_expr_blocks(expr.left, shape_arg_map=shape_arg_map),
                ", ",
                *render_expr_blocks(expr.right, shape_arg_map=shape_arg_map),
                ")",
            ]
        return [
            "(",
            *render_expr_blocks(expr.left, shape_arg_map=shape_arg_map),
            " ",
            op,
            " ",
            *render_expr_blocks(expr.right, shape_arg_map=shape_arg_map),
            ")",
        ]
    if isinstance(expr, FunctionCall):
        result = [expr.function.name, "("]
        for arg in expr.arguments:
            result.extend(render_expr_blocks(arg, shape_arg_map=shape_arg_map))
            result.append(", ")
        if result[-1] == ", ":
            result.pop()
        result.append(")")
        return result
    if isinstance(expr, GetAttr):
        return [*render_expr_blocks(expr.variable, shape_arg_map=shape_arg_map), ".", expr.attr]
    if isinstance(expr, GetItem):
        return _render_getitem_blocks(expr, shape_arg_map=shape_arg_map)
    if isinstance(expr, IntrinsicFunction):
        result = [expr.token, "("]
        for arg in expr.arguments:
            result.extend(render_expr_blocks(arg, shape_arg_map=shape_arg_map))
            result.append(", ")
        if result[-1] == ", ":
            result.pop()
        result.append(")")
        return result
    if isinstance(expr, ArrayConstructor):
        result = ["{"]
        for element in expr.elements:
            if element is None:
                result.append("0")
            else:
                result.extend(render_expr_blocks(element, shape_arg_map=shape_arg_map))
            result.append(", ")
        if result[-1] == ", ":
            result.pop()
        result.append("}")
        return result
    if isinstance(expr, Re):
        return ["creal", "(", *render_expr_blocks(expr.variable, shape_arg_map=shape_arg_map), ")"]
    if isinstance(expr, Im):
        return ["cimag", "(", *render_expr_blocks(expr.variable, shape_arg_map=shape_arg_map), ")"]
    return [str(expr)]


def _render_index_expr(expr: Any, *, shape_arg_map: dict[str, str] | None = None) -> list[str]:
    if isinstance(expr, LiteralNode) and isinstance(expr.value, (int, np.integer)):
        return [str(int(expr.value))]
    if isinstance(expr, (int, np.integer)):
        return [str(int(expr))]
    return render_expr_blocks(expr, shape_arg_map=shape_arg_map)


def _render_getitem_blocks(
    expr: GetItem,
    *,
    shape_arg_map: dict[str, str] | None = None,
) -> list[str]:
    result = render_expr_blocks(expr.variable, shape_arg_map=shape_arg_map)

    def render_block(block: Any) -> list[str]:
        if isinstance(block, slice):
            return _render_slice_blocks(block, shape_arg_map=shape_arg_map)
        return _render_index_expr(block, shape_arg_map=shape_arg_map)

    if isinstance(expr.sliced, tuple):
        dims = []
        for element in expr.sliced:
            dims.append(render_block(element))
        if not expr.variable._shape.fortran_order:
            dims = dims[::-1]
        for dim in dims:
            result += ["[", *dim, "]"]
    else:
        result += ["[", *render_block(expr.sliced), "]"]
    return result


def _render_slice_blocks(
    slice_: slice,
    *,
    shape_arg_map: dict[str, str] | None = None,
) -> list[str]:
    result: list[str] = []
    if slice_.start is not None:
        result += _render_index_expr(slice_.start, shape_arg_map=shape_arg_map)
    result.append(":")
    if slice_.stop is not None:
        stop = slice_.stop - 1 if syntax_settings.c_like_bounds else slice_.stop
        result += _render_index_expr(stop, shape_arg_map=shape_arg_map)
    if slice_.step is not None:
        result.append(":")
        result += _render_index_expr(slice_.step, shape_arg_map=shape_arg_map)
    return result


def render_stmt_lines(stmt: Any, indent: int = 0) -> list[str]:
    if stmt is None:
        return []
    if isinstance(stmt, str):
        return [print_block([stmt], indent=indent)]
    if isinstance(stmt, Comment):
        prefix = getattr(stmt, "prefix", "// ")
        return [print_block(stmt.comment, indent=indent, prefix=prefix)]
    blocks = _render_stmt_blocks(stmt)
    if blocks is not None:
        return [print_block(blocks, indent=indent)]

    if isinstance(stmt, (Do, DoWhile, If, ElseIf, Else, SelectCase, Case, Interface)):
        return _render_scoped_stmt_lines(stmt, indent)
    if isinstance(
        stmt,
        (
            ModuleDeclaration,
            SubroutineDeclaration,
            InterfaceDeclaration,
            FunctionInterfaceDeclaration,
            DerivedTypeDeclaration,
        ),
    ):
        return _render_scoped_stmt_lines(stmt, indent)
    return [print_block(["/* unsupported statement */"], indent=indent)]


def _render_stmt_blocks(stmt: Any) -> list[str] | None:
    if isinstance(stmt, Use):
        result = ["/* use ", stmt.module.name, " */"]
        if stmt.only is not None:
            result = ["/* use ", stmt.module.name, ", only ", stmt.only.name, " */"]
        return result
    if isinstance(stmt, Implicit):
        return ["/* implicit ", stmt.implicit_type, " */"]
    if isinstance(stmt, Assignment):
        return [*render_expr_blocks(stmt.target), " = ", *render_expr_blocks(stmt.value), ";"]
    if isinstance(stmt, SimpleStatement):
        token = getattr(stmt.__class__, "token", "")
        if token:
            return [token, ";"]
        return [";"]
    if isinstance(stmt, Return):
        return ["return;"]
    if isinstance(stmt, Print):
        result = ["printf("]
        for child in stmt.children:
            if isinstance(child, str):
                result.append(f"\"{child.replace('"', '\\"')}\"")
            else:
                result += render_expr_blocks(child)
            result.append(", ")
        if result[-1] == ", ":
            result.pop()
        result.append(");")
        return result
    if isinstance(stmt, Call):
        if isinstance(stmt.function, str):
            result = [stmt.function]
        else:
            result = [stmt.function.name]
        result.append("(")
        for arg in stmt.arguments:
            result += render_expr_blocks(arg)
            result += [", "]
        if result[-1] == ", ":
            result.pop()
        result.append(");")
        return result
    if isinstance(stmt, Allocate):
        return _render_allocate_blocks(stmt)
    if isinstance(stmt, Deallocate):
        return ["free(", *render_expr_blocks(stmt.array), ");"]
    if isinstance(stmt, Contains):
        return ["/* contains */"]
    if isinstance(stmt, PointerAssignment):
        return [
            *render_expr_blocks(stmt.pointer),
            " = ",
            *render_expr_blocks(stmt.target),
            ";",
        ]
    if isinstance(stmt, VariableDeclaration):
        return _render_variable_declaration_blocks(stmt)
    return None


def _render_allocate_blocks(stmt: Allocate) -> list[str]:
    dims = []
    for argument in stmt.shape:
        dims.append([*render_expr_blocks(argument)])

    if not stmt.target._shape.fortran_order:
        dims = dims[::-1]

    dtype = DataType.from_ftype(stmt.target._ftype)
    ctype = dtype.get_cnumpy()
    size_terms = []
    for dim in dims:
        size_terms += dim + [" * "]
    if size_terms:
        size_terms.pop()
    else:
        size_terms = ["1"]

    result = [
        *render_expr_blocks(stmt.target),
        " = (",
        ctype,
        "*)malloc(sizeof(",
        ctype,
        ") * ",
        *size_terms,
        ");",
    ]
    return result


def _render_scoped_stmt_lines(stmt: Any, indent: int) -> list[str]:
    start_blocks = _render_scoped_start_blocks(stmt)
    end_blocks = _render_scoped_end_blocks(stmt)
    result = [print_block(start_blocks, indent=indent)]
    for child in _render_scoped_statements(stmt):
        if isinstance(stmt, If) and isinstance(child, (ElseIf, Else)):
            result.extend(render_stmt_lines(child, indent=indent))
        else:
            result.extend(render_stmt_lines(child, indent=indent + 1))
    if end_blocks:
        result.append(print_block(end_blocks, indent=indent))
    return result


def _render_scoped_start_blocks(stmt: Any) -> list[str]:
    if isinstance(stmt, Do):
        iterator = render_expr_blocks(stmt.iterator)
        start = render_expr_blocks(stmt.start)
        end = render_expr_blocks(stmt.end)
        if stmt.step is not None:
            step = render_expr_blocks(stmt.step)
        else:
            step = ["1"]
        condition = "<="
        if isinstance(stmt.step, LiteralNode) and isinstance(stmt.step.value, (int, float)):
            if stmt.step.value < 0:
                condition = ">="
        return [
            "for (",
            *iterator,
            " = ",
            *start,
            "; ",
            *iterator,
            " ",
            condition,
            " ",
            *end,
            "; ",
            *iterator,
            " += ",
            *step,
            ") {",
        ]
    if isinstance(stmt, DoWhile):
        return ["while (", *render_expr_blocks(stmt.condition), ") {"]
    if isinstance(stmt, If):
        return ["if (", *render_expr_blocks(stmt.condition), ") {"]
    if isinstance(stmt, ElseIf):
        return ["} else if (", *render_expr_blocks(stmt.condition), ") {"]
    if isinstance(stmt, Else):
        return ["} else {"]
    if isinstance(stmt, SelectCase):
        return ["switch (", *render_expr_blocks(stmt.value), ") {"]
    if isinstance(stmt, Case):
        return ["case ", *render_expr_blocks(stmt.value), ":"]
    if isinstance(stmt, Interface):
        return ["/* interface */"]
    if isinstance(stmt, ModuleDeclaration):
        return ["/* module ", stmt.module.name, " */"]
    if isinstance(stmt, SubroutineDeclaration):
        return _render_subroutine_start_blocks(stmt.subroutine)
    if isinstance(stmt, InterfaceDeclaration):
        return _render_subroutine_start_blocks(stmt.subroutine)
    if isinstance(stmt, FunctionInterfaceDeclaration):
        return _render_function_interface_start_blocks(stmt.function)
    if isinstance(stmt, DerivedTypeDeclaration):
        return ["struct ", stmt.derived_type.name, " {"]
    raise NotImplementedError(f"Unsupported scoped statement: {type(stmt)}")


def _render_scoped_end_blocks(stmt: Any) -> list[str]:
    if isinstance(stmt, (Do, DoWhile)):
        return ["}"]
    if isinstance(stmt, If):
        return ["}"]
    if isinstance(stmt, (ElseIf, Else)):
        return []
    if isinstance(stmt, SelectCase):
        return ["}"]
    if isinstance(stmt, Case):
        return []
    if isinstance(stmt, Interface):
        return ["/* end interface */"]
    if isinstance(stmt, ModuleDeclaration):
        return ["/* end module ", stmt.module.name, " */"]
    if isinstance(stmt, SubroutineDeclaration):
        return ["}"]
    if isinstance(stmt, InterfaceDeclaration):
        return ["}"]
    if isinstance(stmt, FunctionInterfaceDeclaration):
        return ["}"]
    if isinstance(stmt, DerivedTypeDeclaration):
        return ["};"]
    raise NotImplementedError(f"Unsupported scoped statement: {type(stmt)}")


def _render_scoped_statements(stmt: Any) -> list[Any]:
    if isinstance(stmt, If):
        result = list(stmt.scope.get_statements())
        for branch in stmt.orelse:
            result.append(branch)
        return result
    if isinstance(stmt, ElseIf):
        return list(stmt.scope.get_statements())
    if isinstance(stmt, Else):
        return list(stmt.scope.get_statements())
    if isinstance(stmt, (Do, DoWhile, SelectCase, Case)):
        return list(stmt.scope.get_statements())
    if isinstance(stmt, Interface):
        return [method.get_interface_declaration() for method in stmt.methods]
    if isinstance(stmt, ModuleDeclaration):
        return list(stmt.get_statements())
    if isinstance(stmt, SubroutineDeclaration):
        return list(stmt.get_statements())
    if isinstance(stmt, InterfaceDeclaration):
        return list(stmt.get_statements())
    if isinstance(stmt, FunctionInterfaceDeclaration):
        return list(stmt.get_statements())
    if isinstance(stmt, DerivedTypeDeclaration):
        return list(stmt.get_statements())
    return []


def _render_subroutine_start_blocks(subroutine: Any) -> list[str]:
    result: list[str] = []

    result.extend(["void", " ", subroutine.name, "("])

    from numeta.ast.module import ExternalModule

    is_external = isinstance(getattr(subroutine, "parent", None), ExternalModule)
    for variable in subroutine.arguments.values():
        if variable.intent is None and not is_external:
            continue
        result.extend(render_expr_blocks(variable))
        result.append(", ")

    if result[-1] == ", ":
        result.pop()
    result.append(") {")

    return result


def _render_function_interface_start_blocks(function: Any) -> list[str]:
    result = ["void", " ", function.name, "("]

    for variable in function.arguments:
        result.extend(render_expr_blocks(variable))
        result.append(", ")

    if result[-1] == ", ":
        result.pop()
    result.append(") {")

    return result


def _render_variable_declaration_blocks(stmt: VariableDeclaration) -> list[str]:
    result = _render_type_blocks(stmt.variable._ftype)

    if stmt.variable.parameter:
        result = ["const ", *result]

    shape = stmt.variable._shape
    if stmt.variable.allocatable or stmt.variable.pointer or shape is UNKNOWN:
        result += [" *", stmt.variable.name]
    elif shape is SCALAR:
        result += [" ", stmt.variable.name]
    elif shape.dims:
        result += [" ", stmt.variable.name]
        result += _render_shape_blocks(shape.dims, fortran_order=shape.fortran_order)

    if stmt.variable.assign is not None:
        if isinstance(stmt.variable.assign, (int, float, complex, bool, str)):
            values = _literal_blocks_from_ftype(stmt.variable.assign, stmt.variable._ftype)
            result += [" = ", *values]
        elif isinstance(stmt.variable.assign, np.ndarray):
            values = []
            for v in stmt.variable.assign.ravel():
                values += _literal_blocks_from_ftype(v, stmt.variable._ftype)
                values.append(", ")
            if values:
                values.pop()
            result += [" = {", *values, "}"]
        else:
            raise ValueError("Can only assign scalars or numpy ndarrays")

        if stmt.variable._shape is UNKNOWN:
            raise ValueError(
                "Cannot assign to a variable with unknown shape. "
                "Please specify the shape of the variable."
            )

    result.append(";")
    return result


def _render_shape_blocks(shape, fortran_order: bool = True) -> list[str]:
    result: list[str] = []

    def convert(element):
        if element is None:
            return ["[", "*", "]"]
        if isinstance(element, int):
            return ["[", str(element), "]"]
        if isinstance(element, slice):
            start = (
                element.start if element.start is not None else syntax_settings.array_lower_bound
            )
            stop = element.stop if element.stop is not None else ""
            if syntax_settings.c_like_bounds and isinstance(stop, int):
                stop = stop - 1
            if element.step is not None:
                raise NotImplementedError("Step in array dimensions is not implemented yet")
            return ["[", str(start), ":", str(stop), "]"]
        return ["[", *render_expr_blocks(element), "]"]

    if isinstance(shape, tuple):
        dims = [convert(d) for d in shape]
        if not fortran_order:
            dims = dims[::-1]
        for dim in dims:
            result += dim
    else:
        result += convert(shape)

    return result


def _render_type_blocks(ftype: FortranType) -> list[str]:
    dtype = DataType.from_ftype(ftype)
    return [dtype.get_cnumpy()]
