from __future__ import annotations

from typing import Any

import numpy as np

from numeta.array_shape import ArrayShape, SCALAR, UNKNOWN
from numeta.fortran.settings import settings as syntax_settings
from numeta.ast.statements.tools import print_block

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


def _literal_blocks_from_ftype(value: Any, ftype: FortranType) -> list[str]:
    kind = ftype.get_kind_str()

    if ftype.type == "type":
        return [f"{value}"]
    if ftype.type == "integer":
        return [f"{int(value)}_{kind}"]
    if ftype.type == "real":
        return [f"{float(value)}_{kind}"]
    if ftype.type == "complex":
        return ["(", f"{value.real}_{kind}", ",", f"{value.imag}_{kind}", ")"]
    if ftype.type == "logical":
        return [f".true._{kind}" if value is True else f".false._{kind}"]
    if ftype.type == "character":
        return [f'"{value}"']
    raise ValueError(f"Unknown type: {ftype.type}")


def render_expr_blocks(expr: Any) -> list[str]:
    if expr is None:
        return [""]
    if isinstance(expr, LiteralNode):
        return _literal_blocks_from_ftype(expr.value, expr._ftype)
    if isinstance(expr, (int, float, complex, bool, str, np.generic)):
        literal = LiteralNode(expr)
        return _literal_blocks_from_ftype(literal.value, literal._ftype)
    if isinstance(expr, Variable):
        return [expr.name]
    if isinstance(expr, NamedEntity):
        return [expr.name]
    if isinstance(expr, BinaryOperationNodeNoPar):
        return [
            *render_expr_blocks(expr.left),
            expr.op,
            *render_expr_blocks(expr.right),
        ]
    if isinstance(expr, BinaryOperationNode):
        return [
            "(",
            *render_expr_blocks(expr.left),
            expr.op,
            *render_expr_blocks(expr.right),
            ")",
        ]
    if isinstance(expr, FunctionCall):
        result = [expr.function.name, "("]
        for arg in expr.arguments:
            result.extend(render_expr_blocks(arg))
            result.append(", ")
        if result[-1] == ", ":
            result.pop()
        result.append(")")
        return result
    if isinstance(expr, GetAttr):
        return [*render_expr_blocks(expr.variable), "%", expr.attr]
    if isinstance(expr, GetItem):
        return _render_getitem_blocks(expr)
    if isinstance(expr, IntrinsicFunction):
        result = [expr.token, "("]
        for arg in expr.arguments:
            result.extend(render_expr_blocks(arg))
            result.append(", ")
        if result[-1] == ", ":
            result.pop()
        result.append(")")
        return result
    if isinstance(expr, ArrayConstructor):
        result = ["["]
        for element in expr.elements:
            if element is None:
                result.append("None")
            else:
                result.extend(render_expr_blocks(element))
            result.append(", ")
        if result[-1] == ", ":
            result.pop()
        result.append("]")
        return result
    if isinstance(expr, Re):
        return [*render_expr_blocks(expr.variable), "%", "re"]
    if isinstance(expr, Im):
        return [*render_expr_blocks(expr.variable), "%", "im"]
    return [str(expr)]


def _render_index_expr(expr: Any) -> list[str]:
    if isinstance(expr, LiteralNode) and isinstance(expr.value, (int, np.integer)):
        return [str(int(expr.value))]
    if isinstance(expr, (int, np.integer)):
        return [str(int(expr))]
    return render_expr_blocks(expr)


def _render_getitem_blocks(expr: GetItem) -> list[str]:
    result = render_expr_blocks(expr.variable)

    def render_block(block: Any) -> list[str]:
        if isinstance(block, slice):
            return _render_slice_blocks(block)
        return _render_index_expr(block)

    result.append("(")
    if isinstance(expr.sliced, tuple):
        dims = []
        for element in expr.sliced:
            dims.append(render_block(element))
        if not expr.variable._shape.fortran_order:
            dims = dims[::-1]
        result += dims[0]
        for dim in dims[1:]:
            result += [",", " "] + dim
    else:
        result += render_block(expr.sliced)
    result.append(")")
    return result


def _render_slice_blocks(slice_: slice) -> list[str]:
    result: list[str] = []
    if slice_.start is not None:
        result += _render_index_expr(slice_.start)
    result.append(":")
    if slice_.stop is not None:
        stop = slice_.stop - 1 if syntax_settings.c_like_bounds else slice_.stop
        result += _render_index_expr(stop)
    if slice_.step is not None:
        result.append(":")
        result += _render_index_expr(slice_.step)
    return result


def render_stmt_lines(stmt: Any, indent: int = 0) -> list[str]:
    if stmt is None:
        return []
    if isinstance(stmt, str):
        return [print_block([stmt], indent=indent)]
    if isinstance(stmt, Comment):
        prefix = getattr(stmt, "prefix", "! ")
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
    return [print_block(["! unsupported statement"], indent=indent)]


def _render_stmt_blocks(stmt: Any) -> list[str] | None:
    if isinstance(stmt, Use):
        result = ["use", " ", stmt.module.name]
        if stmt.only is not None:
            result += [", ", "only", ": ", stmt.only.name]
        return result
    if isinstance(stmt, Implicit):
        return ["implicit", " ", stmt.implicit_type]
    if isinstance(stmt, Assignment):
        return [*render_expr_blocks(stmt.target), "=", *render_expr_blocks(stmt.value)]
    if isinstance(stmt, SimpleStatement):
        token = getattr(stmt.__class__, "token", "")
        return [token]
    if isinstance(stmt, Return):
        return ["return"]
    if isinstance(stmt, Print):
        result = ["print *, "]
        for child in stmt.children:
            if isinstance(child, str):
                result.append(f"\"{child.replace('"', '""')}\"")
            else:
                result += render_expr_blocks(child)
            result.append(", ")
        if result[-1] == ", ":
            result.pop()
        return result
    if isinstance(stmt, Call):
        if isinstance(stmt.function, str):
            result = ["call", " ", stmt.function]
        else:
            result = ["call", " ", stmt.function.name]
        result.append("(")
        for arg in stmt.arguments:
            result += render_expr_blocks(arg)
            result += [", "]
        if result[-1] == ", ":
            result.pop()
        result.append(")")
        return result
    if isinstance(stmt, Allocate):
        return _render_allocate_blocks(stmt)
    if isinstance(stmt, Deallocate):
        return ["deallocate", "(", *render_expr_blocks(stmt.array), ")"]
    if isinstance(stmt, Contains):
        return ["contains"]
    if isinstance(stmt, PointerAssignment):
        return [
            *render_expr_blocks(stmt.pointer),
            *_render_shape_blocks(stmt.pointer_shape, fortran_order=True),
            "=>",
            *render_expr_blocks(stmt.target),
        ]
    if isinstance(stmt, VariableDeclaration):
        return _render_variable_declaration_blocks(stmt)
    return None


def _render_allocate_blocks(stmt: Allocate) -> list[str]:
    result = ["allocate", "("]
    result += render_expr_blocks(stmt.target)

    dims = []
    for argument in stmt.shape:
        if (lbound := syntax_settings.array_lower_bound) != 1:
            dims.append([str(lbound), ":", *render_expr_blocks(argument + (lbound - 1))])
        else:
            dims.append([*render_expr_blocks(argument)])

    if not stmt.target._shape.fortran_order:
        dims = dims[::-1]

    result.append("(")
    result += dims[0]
    for dim in dims[1:]:
        result += [",", " "] + dim
    result.append(")")

    result.append(")")
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
        result = ["do", " "]
        result += render_expr_blocks(stmt.iterator)
        result += [" ", "=", " "]
        result += render_expr_blocks(stmt.start)
        result.append(", ")
        result += render_expr_blocks(stmt.end)
        if stmt.step is not None:
            result.append(", ")
            result += render_expr_blocks(stmt.step)
        return result
    if isinstance(stmt, DoWhile):
        return ["do while", " ", "(", *render_expr_blocks(stmt.condition), ")"]
    if isinstance(stmt, If):
        return ["if", "(", *render_expr_blocks(stmt.condition), ")", "then"]
    if isinstance(stmt, ElseIf):
        return ["elseif", "(", *render_expr_blocks(stmt.condition), ")", "then"]
    if isinstance(stmt, Else):
        return ["else"]
    if isinstance(stmt, SelectCase):
        return ["select", " ", "case", " ", "(", *render_expr_blocks(stmt.value), ")"]
    if isinstance(stmt, Case):
        return ["case", " ", "(", *render_expr_blocks(stmt.value), ")"]
    if isinstance(stmt, Interface):
        return ["interface"]
    if isinstance(stmt, ModuleDeclaration):
        return ["module", " ", stmt.module.name]
    if isinstance(stmt, SubroutineDeclaration):
        return _render_subroutine_start_blocks(stmt.subroutine)
    if isinstance(stmt, InterfaceDeclaration):
        return _render_subroutine_start_blocks(stmt.subroutine)
    if isinstance(stmt, FunctionInterfaceDeclaration):
        return _render_function_interface_start_blocks(stmt.function)
    if isinstance(stmt, DerivedTypeDeclaration):
        if syntax_settings.derived_type_bind_c:
            return ["type", ", ", "bind(C)", " ", "::", " ", stmt.derived_type.name]
        return ["type", " ", "::", " ", stmt.derived_type.name]
    raise NotImplementedError(f"Unsupported scoped statement: {type(stmt)}")


def _render_scoped_end_blocks(stmt: Any) -> list[str]:
    if isinstance(stmt, (Do, DoWhile)):
        return ["end", " ", "do"]
    if isinstance(stmt, If):
        return ["end", " ", "if"]
    if isinstance(stmt, (ElseIf, Else)):
        return []
    if isinstance(stmt, SelectCase):
        return ["end", " ", "select"]
    if isinstance(stmt, Case):
        return []
    if isinstance(stmt, Interface):
        return ["end", " ", "interface"]
    if isinstance(stmt, ModuleDeclaration):
        return ["end", " ", "module", " ", stmt.module.name]
    if isinstance(stmt, SubroutineDeclaration):
        return ["end", " ", "subroutine", " ", stmt.subroutine.name]
    if isinstance(stmt, InterfaceDeclaration):
        return ["end", " ", "subroutine", " ", stmt.subroutine.name]
    if isinstance(stmt, FunctionInterfaceDeclaration):
        return ["end", " ", "function", " ", stmt.function.name]
    if isinstance(stmt, DerivedTypeDeclaration):
        return ["end", " ", "type", " ", stmt.derived_type.name]
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

    if getattr(subroutine, "pure", False):
        result += ["pure", " "]
    if getattr(subroutine, "elemental", False):
        result += ["elemental", " "]

    result.extend(["subroutine", " ", subroutine.name, "("])

    from numeta.ast.module import ExternalModule

    is_external = isinstance(getattr(subroutine, "parent", None), ExternalModule)
    for variable in subroutine.arguments.values():
        if variable.intent is None and not is_external:
            continue
        result.extend(render_expr_blocks(variable))
        result.append(", ")

    if result[-1] == ", ":
        result.pop()
    result.append(")")

    if getattr(subroutine, "bind_c", False):
        result.extend([" ", f"bind(C, name='{subroutine.name}')"])

    return result


def _render_function_interface_start_blocks(function: Any) -> list[str]:
    result = ["function", " ", function.name, "("]

    for variable in function.arguments:
        result.extend(render_expr_blocks(variable))
        result.append(", ")

    if result[-1] == ", ":
        result.pop()
    result.append(")")

    result_variable = function.get_result_variable()
    result.extend([" ", "result", "(", result_variable.name, ")"])

    if getattr(function, "bind_c", False):
        result.extend([" ", f"bind(C, name='{function.name}')"])

    return result


def _render_variable_declaration_blocks(stmt: VariableDeclaration) -> list[str]:
    result = _render_type_blocks(stmt.variable._ftype)

    if stmt.variable.allocatable:
        result += [", ", "allocatable", ", ", "dimension"]
        result += ["("] + [":", ","] * (len(stmt.variable._shape.dims) - 1) + [":", ")"]
    elif stmt.variable.pointer:
        result += [", ", "pointer"]
        if stmt.variable._shape is not SCALAR:
            result += [", ", "dimension"]
            result += ["("] + [":", ","] * (len(stmt.variable._shape.dims) - 1) + [":", ")"]
    elif stmt.variable._shape is UNKNOWN:
        result += [", ", "dimension", "(", str(syntax_settings.array_lower_bound), ":", "*", ")"]
    elif stmt.variable._shape.dims:
        result += [", ", "dimension"]
        result += _render_shape_blocks(
            stmt.variable._shape.dims, fortran_order=stmt.variable._shape.fortran_order
        )

    if stmt.variable.intent is not None:
        result += [", ", "intent", "(", stmt.variable.intent, ")"]

    if syntax_settings.force_value:
        if stmt.variable._shape is SCALAR and stmt.variable.intent == "in":
            result += [", ", "value"]

    if stmt.variable.parameter:
        result += [", ", "parameter"]

    if stmt.variable.target:
        result += [", ", "contiguous" if stmt.variable.pointer else "target"]

    if stmt.variable.bind_c:
        result += [", ", "bind", "(", "C", ", ", "name=", "'", stmt.variable.name, "'", ")"]

    result += [" :: ", stmt.variable.name]

    if stmt.variable.assign is not None:
        if isinstance(stmt.variable.assign, (int, float, complex, bool, str)):
            values = _literal_blocks_from_ftype(stmt.variable.assign, stmt.variable._ftype)
        elif isinstance(stmt.variable.assign, np.ndarray):
            values = []
            for v in stmt.variable.assign.ravel():
                values += _literal_blocks_from_ftype(v, stmt.variable._ftype)
                values.append(", ")
            if values:
                values.pop()
        else:
            raise ValueError("Can only assign scalars or numpy ndarrays")

        if stmt.variable._shape is UNKNOWN:
            raise ValueError(
                "Cannot assign to a variable with unknown shape. "
                "Please specify the shape of the variable."
            )
        result += [";", " data ", stmt.variable.name, " / ", *values, " /"]

    return result


def _render_shape_blocks(shape, fortran_order: bool = True) -> list[str]:
    lbound = syntax_settings.array_lower_bound
    result = ["("]

    def convert(element):
        if element is None:
            return [str(lbound), ":", "*"]
        if isinstance(element, int):
            return [str(lbound), ":", str(element + (lbound - 1))]
        if isinstance(element, slice):
            shift_end = None
            if element.start is None:
                start = [str(lbound)]
                shift_end = lbound - 1
            elif isinstance(element.start, int):
                start = [str(element.start)]
            else:
                start = render_expr_blocks(element.start)

            if element.stop is None:
                stop = [""]
            elif isinstance(element.stop, int):
                stop = (
                    [str(element.stop + shift_end)]
                    if shift_end is not None
                    else [str(element.stop)]
                )
            else:
                if shift_end is not None:
                    stop = render_expr_blocks(element.stop + shift_end)
                else:
                    stop = render_expr_blocks(element.stop)

            if element.step is not None:
                raise NotImplementedError("Step in array dimensions is not implemented yet")
            return start + [":"] + stop
        return [str(lbound), ":", *render_expr_blocks(element + (lbound - 1))]

    if isinstance(shape, tuple):
        dims = [convert(d) for d in shape]
        if not fortran_order:
            dims = dims[::-1]
        result += dims[0]
        for dim in dims[1:]:
            result += [",", " "] + dim
    else:
        result += convert(shape)

    result += [")"]
    return result


def _render_type_blocks(ftype: FortranType) -> list[str]:
    kind = ftype.get_kind_str()
    if ftype.kind is not None:
        return [ftype.type, "(", kind, ")"]
    return [ftype.type]
