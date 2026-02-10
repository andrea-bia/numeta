from __future__ import annotations

from typing import Any

import numpy as np

from numeta.settings import settings
from numeta.ast.statements.tools import print_block

syntax_settings = settings.syntax

from .fortran_syntax import render_expr_blocks, render_stmt_lines

from numeta.ir.nodes import (
    IRAllocate,
    IRAssign,
    IRBinary,
    IRCall,
    IRCallExpr,
    IRDeallocate,
    IRExpr,
    IRFor,
    IRGetAttr,
    IRGetItem,
    IRIf,
    IRIntrinsic,
    IRLiteral,
    IRProcedure,
    IRReturn,
    IRSlice,
    IRUnary,
    IRVarRef,
    IRWhile,
    IROpaqueExpr,
    IROpaqueStmt,
)


_FORTRAN_BINARY_OPS = {
    "eq": ".eq.",
    "ne": ".ne.",
    "lt": ".lt.",
    "le": ".le.",
    "gt": ".gt.",
    "ge": ".ge.",
    "and": ".and.",
    "or": ".or.",
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "pow": "**",
}

# Mapping for intrinsics that differ in Fortran
_FORTRAN_INTRINSIC_MAP = {
    "ceil": "ceiling",
    "copysign": "sign",
}


class FortranEmitter:
    def emit_procedure(self, proc: IRProcedure) -> str:
        pure = bool(proc.metadata.get("fortran_pure", False))
        elemental = bool(proc.metadata.get("fortran_elemental", False))
        bind_c = bool(proc.metadata.get("fortran_bind_c", False))

        args = [arg.name for arg in proc.args]

        start_blocks: list[str] = []
        if pure:
            start_blocks += ["pure", " "]
        if elemental:
            start_blocks += ["elemental", " "]
        start_blocks += ["subroutine", " ", proc.name, "("]
        for name in args:
            start_blocks += [name, ", "]
        if args:
            start_blocks.pop()
        start_blocks += [")"]
        if bind_c:
            start_blocks += [" ", f"bind(C, name='{proc.name}')"]

        lines: list[str] = []
        lines.append(print_block(start_blocks, indent=0))

        prelude_lines = proc.metadata.get("fortran_prelude_lines", [])
        prelude_items = proc.metadata.get("fortran_prelude_items", [])
        if prelude_lines:
            lines.extend(prelude_lines)
        elif prelude_items:
            for stmt in prelude_items:
                lines.extend(render_stmt_lines(stmt, indent=1))
        else:
            lines.append(print_block(["implicit", " ", "none"], indent=1))

        for var in proc.args:
            lines.append(self._declare_variable(var, indent=1))
        for var in proc.locals:
            lines.append(self._declare_variable(var, indent=1))

        for stmt in proc.body:
            lines.extend(self._emit_stmt(stmt, indent=1))

        lines.append(print_block(["end", " ", "subroutine", " ", proc.name], indent=0))
        return "".join(lines)

    def _emit_stmt(self, stmt, *, indent: int) -> list[str]:
        if isinstance(stmt, IRAssign):
            blocks = self._expr_blocks(stmt.target) + ["="] + self._expr_blocks(stmt.value)
            return [print_block(blocks, indent=indent)]
        if isinstance(stmt, IRCall):
            blocks = ["call", " "] + self._expr_blocks(stmt.func) + ["("]
            blocks += self._join_args(stmt.args)
            blocks += [")"]
            return [print_block(blocks, indent=indent)]
        if isinstance(stmt, IRIf):
            lines = [
                print_block(["if", "(", *self._expr_blocks(stmt.cond), ")", "then"], indent=indent)
            ]
            for child in stmt.then:
                lines.extend(self._emit_stmt(child, indent=indent + 1))

            if stmt.else_:
                for idx, branch in enumerate(stmt.else_):
                    if (
                        isinstance(branch, IRIf)
                        and getattr(branch.source, "__class__", None) is not None
                    ):
                        if branch.source.__class__.__name__ == "ElseIf":
                            lines.append(
                                print_block(
                                    ["elseif", "(", *self._expr_blocks(branch.cond), ")", "then"],
                                    indent=indent,
                                )
                            )
                            for child in branch.then:
                                lines.extend(self._emit_stmt(child, indent=indent + 1))
                            continue
                    lines.append(print_block(["else"], indent=indent))
                    remaining = stmt.else_[idx:]
                    for child in remaining:
                        lines.extend(self._emit_stmt(child, indent=indent + 1))
                    break

            lines.append(print_block(["end", " ", "if"], indent=indent))
            return lines
        if isinstance(stmt, IRFor):
            var = stmt.var
            var_name = var.name if var is not None else "<var>"
            blocks = ["do", " ", var_name, " ", "=", " "]
            blocks += self._expr_blocks(stmt.start)
            blocks.append(", ")
            blocks += self._expr_blocks(stmt.stop)
            if stmt.step is not None:
                blocks.append(", ")
                blocks += self._expr_blocks(stmt.step)
            lines = [print_block(blocks, indent=indent)]
            for child in stmt.body:
                lines.extend(self._emit_stmt(child, indent=indent + 1))
            lines.append(print_block(["end", " ", "do"], indent=indent))
            return lines
        if isinstance(stmt, IRWhile):
            blocks = ["do while", " ", "("] + self._expr_blocks(stmt.cond) + [")"]
            lines = [print_block(blocks, indent=indent)]
            for child in stmt.body:
                lines.extend(self._emit_stmt(child, indent=indent + 1))
            lines.append(print_block(["end", " ", "do"], indent=indent))
            return lines
        if isinstance(stmt, IRAllocate):
            blocks = ["allocate", "("]
            var_blocks = self._expr_blocks(stmt.var)
            dims = self._format_allocate_dims(stmt.var, stmt.dims)
            blocks += var_blocks + dims + [")"]
            return [print_block(blocks, indent=indent)]
        if isinstance(stmt, IRDeallocate):
            blocks = ["deallocate", "("] + self._expr_blocks(stmt.var) + [")"]
            return [print_block(blocks, indent=indent)]
        if isinstance(stmt, IRReturn):
            if stmt.value is None:
                return [print_block(["return"], indent=indent)]
            return [print_block(["return", " ", *self._expr_blocks(stmt.value)], indent=indent)]
        if isinstance(stmt, IROpaqueStmt):
            if stmt.payload is not None:
                return render_stmt_lines(stmt.payload, indent=indent)
        return [print_block(["! unsupported statement"], indent=indent)]

    def _is_integer_expr(self, expr: IRExpr) -> bool:
        if isinstance(expr, IRLiteral) and isinstance(expr.value, int):
            return True
        if expr.vtype and expr.vtype.dtype.name == "integer":
            return True
        if (
            isinstance(expr, IRVarRef)
            and expr.var
            and expr.var.vtype
            and expr.var.vtype.dtype.name == "integer"
        ):
            return True
        return False

    def _expr_blocks(self, expr: IRExpr | None) -> list[str]:
        if expr is None:
            return [""]
        if isinstance(expr, IRLiteral):
            source: Any = expr.source
            if source is not None:
                return render_expr_blocks(source)
            if isinstance(expr.value, str):
                return [f'"{expr.value}"']
            if isinstance(expr.value, bool):
                return [".true." if expr.value else ".false."]
            return [str(expr.value)]
        if isinstance(expr, IRVarRef):
            var = expr.var
            if var is not None:
                return [var.name]
            return ["<var>"]
        if isinstance(expr, IRBinary):
            op = _FORTRAN_BINARY_OPS.get(expr.op, expr.op)
            return ["(", *self._expr_blocks(expr.left), op, *self._expr_blocks(expr.right), ")"]
        if isinstance(expr, IRUnary):
            if expr.op == "neg":
                return ["-", "(", *self._expr_blocks(expr.operand), ")"]
            if expr.op == "not":
                return [".not.", "(", *self._expr_blocks(expr.operand), ")"]
            return [expr.op, "(", *self._expr_blocks(expr.operand), ")"]
        if isinstance(expr, IRCallExpr):
            blocks = self._expr_blocks(expr.callee) + ["("]
            blocks += self._join_args(expr.args)
            blocks += [")"]
            return blocks
        if isinstance(expr, IRGetAttr):
            return [*self._expr_blocks(expr.base), "%", expr.name]
        if isinstance(expr, IRGetItem):
            base_blocks = self._expr_blocks(expr.base)
            indices = list(expr.indices)
            order = None
            if expr.base is not None and expr.base.vtype is not None:
                shape = expr.base.vtype.shape
                if shape is not None:
                    order = shape.order
            if order == "C" and len(indices) > 1:
                indices = list(reversed(indices))
            dim_blocks = []
            for index in indices:
                if isinstance(index, IRSlice):
                    dim_blocks.append(self._slice_blocks(index))
                else:
                    dim_blocks.append(self._index_blocks(index))
            blocks = base_blocks + ["("]
            if dim_blocks:
                blocks += dim_blocks[0]
                for dim in dim_blocks[1:]:
                    blocks += [",", " "] + dim
            blocks += [")"]
            return blocks
        if isinstance(expr, IRIntrinsic):
            if expr.name == "array_constructor":
                blocks = ["["]
                blocks += self._join_args(expr.args)
                blocks += ["]"]
                return blocks

            # Map intrinsic name if necessary
            name = _FORTRAN_INTRINSIC_MAP.get(expr.name, expr.name)

            # Workaround for LOG10(complex) which is not standard in Fortran
            if name == "log10" and expr.args:
                arg0 = expr.args[0]
                is_complex = False
                if arg0.vtype and arg0.vtype.dtype.name == "complex":
                    is_complex = True
                elif (
                    isinstance(arg0, IRVarRef)
                    and arg0.var
                    and arg0.var.vtype
                    and arg0.var.vtype.dtype.name == "complex"
                ):
                    is_complex = True

                if is_complex:
                    arg_blocks = self._expr_blocks(arg0)
                    return [
                        "(",
                        "log",
                        "(",
                        *arg_blocks,
                        ")",
                        "/",
                        "log",
                        "(",
                        "10.0",
                        ")",
                        ")",
                    ]

            # Cast integer arguments to real for math functions that require it
            if name in {
                "exp",
                "log",
                "log10",
                "sqrt",
                "sin",
                "cos",
                "tan",
                "asin",
                "acos",
                "atan",
                "atan2",
                "sinh",
                "cosh",
                "tanh",
                "asinh",
                "acosh",
                "atanh",
                "erf",
                "erfc",
                "gamma",
                "lgamma",
                "hypot",
            }:
                args_blocks = []
                for arg in expr.args:
                    arg_blocks = self._expr_blocks(arg)
                    if self._is_integer_expr(arg):
                        args_blocks.append(["real(", *arg_blocks, ")"])
                    else:
                        args_blocks.append(arg_blocks)

                blocks = [name, "("]
                for i, arg_b in enumerate(args_blocks):
                    if i > 0:
                        blocks.append(", ")
                    blocks += arg_b
                blocks.append(")")
                return blocks

            blocks = [name, "("]
            blocks += self._join_args(expr.args)
            blocks += [")"]
            return blocks
        if isinstance(expr, IROpaqueExpr):
            if expr.payload is not None:
                return render_expr_blocks(expr.payload)
            return ["<expr>"]
        return ["<expr>"]

    def _join_args(self, args: list[IRExpr]) -> list[str]:
        blocks: list[str] = []
        for arg in args:
            blocks += self._expr_blocks(arg)
            blocks += [", "]
        if blocks:
            blocks.pop()
        return blocks

    def _index_blocks(self, expr: IRExpr) -> list[str]:
        shift = syntax_settings.array_lower_bound
        return self._shift_expr_blocks(expr, shift)

    def _slice_blocks(self, slice_: IRSlice) -> list[str]:
        lbound = syntax_settings.array_lower_bound

        if slice_.start is None:
            start_blocks = [str(lbound)]
        else:
            start_blocks = self._shift_expr_blocks(slice_.start, lbound)

        stop_blocks: list[str] = []
        if slice_.stop is not None:
            stop_blocks = self._shift_expr_blocks(slice_.stop, lbound - 1)

        result = [*start_blocks, ":"]
        if stop_blocks:
            result += stop_blocks
        if slice_.step is not None:
            result.append(":")
            result += self._expr_blocks(slice_.step)
        return result

    def _shift_expr_blocks(self, expr: IRExpr, shift: int) -> list[str]:
        if shift == 0:
            return self._expr_blocks(expr)
        if isinstance(expr, IRLiteral) and isinstance(expr.value, (int, float)):
            return [str(expr.value + shift)]
        op = "+" if shift > 0 else "-"
        return ["(", *self._expr_blocks(expr), op, str(abs(shift)), ")"]

    def _format_allocate_dims(self, var_expr: IRExpr | None, dims: list[IRExpr]) -> list[str]:
        if not dims:
            return []
        order = None
        if var_expr is not None and var_expr.vtype is not None:
            shape = var_expr.vtype.shape
            if shape is not None:
                order = shape.order

        lbound = syntax_settings.array_lower_bound
        dim_blocks = []
        for dim in dims:
            if lbound != 1:
                dim_blocks.append(
                    [
                        str(lbound),
                        ":",
                        *self._shift_expr_blocks(dim, lbound - 1),
                    ]
                )
            else:
                dim_blocks.append(self._expr_blocks(dim))

        if order == "C" and len(dim_blocks) > 1:
            dim_blocks = list(reversed(dim_blocks))

        blocks = ["("] + dim_blocks[0]
        for dim in dim_blocks[1:]:
            blocks += [",", " "] + dim
        blocks.append(")")
        return blocks

    def _declare_variable(self, var, *, indent: int) -> str:
        if var.vtype is None:
            return print_block(["! undeclared :: ", var.name], indent=indent)

        dtype = var.vtype.dtype
        if dtype.kind is not None:
            type_block = [dtype.name, "(", str(dtype.kind), ")"]
        else:
            type_block = [dtype.name]

        blocks = list(type_block)
        shape = var.vtype.shape
        if var.allocatable:
            rank = 1
            if shape is not None and shape.rank:
                rank = shape.rank
            blocks += [", ", "allocatable", ", ", "dimension", "("]
            blocks += [":", ","] * (rank - 1)
            blocks += [":", ")"]
        elif var.pointer:
            blocks += [", ", "pointer"]
            if shape is not None and shape.rank:
                blocks += [", ", "dimension", "("]
                blocks += [":", ","] * (shape.rank - 1)
                blocks += [":", ")"]
        elif shape is not None and shape.dims is None:
            blocks += [
                ", ",
                "dimension",
                "(",
                str(syntax_settings.array_lower_bound),
                ":",
                "*",
                ")",
            ]
        elif shape is not None and shape.rank:
            blocks += [", ", "dimension"]
            blocks += self._shape_blocks(shape)

        if var.intent is not None:
            blocks += [", ", "intent", "(", var.intent, ")"]

        if syntax_settings.force_value:
            if (shape is None or shape.rank == 0) and var.intent == "in":
                blocks += [", ", "value"]

        if var.parameter:
            blocks += [", ", "parameter"]

        if var.target:
            if var.pointer:
                blocks += [", ", "contiguous"]
            else:
                blocks += [", ", "target"]

        if var.bind_c:
            blocks += [", ", "bind", "(", "C", ", ", "name=", "'", var.name, "'", ")"]

        blocks += [" :: ", var.name]

        if var.assign is not None:
            values = []
            if isinstance(var.assign, (int, float, complex, bool, str)):
                values = self._literal_blocks(var.assign)
            elif isinstance(var.assign, np.ndarray):
                for v in var.assign.ravel():
                    values += self._literal_blocks(v)
                    values.append(", ")
                if values:
                    values.pop()
            if values:
                blocks += [";", " data ", var.name, " / ", *values, " /"]

        return print_block(blocks, indent=indent)

    def _shape_blocks(self, shape) -> list[str]:
        lbound = syntax_settings.array_lower_bound
        dims = list(shape.dims or [])
        if shape.order == "C" and len(dims) > 1:
            dims = list(reversed(dims))

        def dim_blocks(dim) -> list[str]:
            if dim is None:
                return [str(lbound), ":", "*"]
            if isinstance(dim, int):
                upper = dim + (lbound - 1)
                return [str(lbound), ":", str(upper)]
            if isinstance(dim, IRExpr):
                blocks = self._expr_blocks(dim)
            else:
                blocks = render_expr_blocks(dim)
                if blocks is None:
                    blocks = [str(dim)]
            if lbound != 1:
                shift = lbound - 1
                op = "+" if shift >= 0 else "-"
                return [str(lbound), ":", "(", *blocks, op, str(abs(shift)), ")"]
            return [str(lbound), ":", *blocks]

        result = ["("]
        for i, dim in enumerate(dims):
            if i:
                result += [",", " "]
            result += dim_blocks(dim)
        result.append(")")
        return result

    def _literal_blocks(self, value) -> list[str]:
        if isinstance(value, str):
            return [f'"{value}"']
        if isinstance(value, bool):
            return [".true." if value else ".false."]
        if isinstance(value, complex):
            return ["(", str(value.real), ",", str(value.imag), ")"]
        return [str(value)]
