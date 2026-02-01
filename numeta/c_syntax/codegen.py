from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from numeta.array_shape import SCALAR
from numeta.datatype import DataType
from numeta.syntax import Subroutine, Variable
from numeta.syntax.expressions import (
    BinaryOperationNode,
    FunctionCall,
    IntrinsicFunction,
    LiteralNode,
)
from numeta.syntax.expressions.getattr import GetAttr
from numeta.syntax.expressions.getitem import GetItem
from numeta.syntax.statements import (
    Assignment,
    Call,
    Do,
    Else,
    ElseIf,
    If,
    Return,
)


_C_OPS = {
    ".eq.": "==",
    ".ne.": "!=",
    ".lt.": "<",
    ".le.": "<=",
    ".gt.": ">",
    ".ge.": ">=",
    ".and.": "&&",
    ".or.": "||",
    "**": "pow",
}


@dataclass(frozen=True)
class CArg:
    name: str
    ctype: str
    is_pointer: bool
    is_const: bool

    def render(self) -> str:
        const_prefix = "const " if self.is_const else ""
        pointer_suffix = "*" if self.is_pointer else ""
        return f"{const_prefix}{self.ctype} {pointer_suffix}{self.name}"


class CCodegen:
    def __init__(self, subroutine: Subroutine) -> None:
        self.subroutine = subroutine
        self._pointer_args: dict[str, bool] = {}
        self._arg_specs: list[CArg] = []
        self._requires_math = False
        self._build_signature()

    @property
    def requires_math(self) -> bool:
        return self._requires_math

    def render(self) -> str:
        lines: list[str] = []
        lines.append("#include <numpy/arrayobject.h>\n")
        if self._requires_math:
            lines.append("#include <math.h>\n")
        lines.append("\n")
        lines.extend(self._render_function())
        return "".join(lines)

    def _build_signature(self) -> None:
        for variable in self.subroutine.arguments.values():
            self._validate_scalar(variable)
            dtype = DataType.from_ftype(variable._ftype)
            ctype = dtype.get_cnumpy()
            is_const = variable.intent == "in"
            is_pointer = self._is_pointer_arg(variable)
            self._pointer_args[variable.name] = is_pointer
            self._arg_specs.append(CArg(variable.name, ctype, is_pointer, is_const))

    def _render_function(self) -> list[str]:
        args = ", ".join(arg.render() for arg in self._arg_specs)
        header = f"void {self.subroutine.name}({args}) {{\n"
        lines = [header]
        lines.extend(self._render_local_declarations(indent=1))
        lines.extend(self._render_statements(self.subroutine.scope.get_statements(), indent=1))
        lines.append("}\n")
        return lines

    def _render_local_declarations(self, indent: int) -> list[str]:
        declarations: list[str] = []
        for variable in self.subroutine.get_local_variables().values():
            self._validate_scalar(variable)
            dtype = DataType.from_ftype(variable._ftype)
            ctype = dtype.get_cnumpy()
            const_prefix = "const " if variable.parameter else ""
            init = ""
            if variable.assign is not None:
                init_value = self._render_literal(variable.assign)
                init = f" = {init_value}"
            declarations.append(f"{'    ' * indent}{const_prefix}{ctype} {variable.name}{init};\n")
        if declarations:
            declarations.append("\n")
        return declarations

    def _render_statements(self, statements: Iterable, indent: int) -> list[str]:
        lines: list[str] = []
        for stmt in statements:
            lines.extend(self._render_statement(stmt, indent=indent))
        return lines

    def _render_statement(self, stmt, indent: int) -> list[str]:
        if isinstance(stmt, Assignment):
            return [f"{'    ' * indent}{self._render_assignment(stmt)}\n"]
        if isinstance(stmt, If):
            return self._render_if(stmt, indent)
        if isinstance(stmt, Do):
            return self._render_do(stmt, indent)
        if isinstance(stmt, Return):
            return [f"{'    ' * indent}return;\n"]
        if isinstance(stmt, Call):
            return [f"{'    ' * indent}{self._render_call(stmt)}\n"]
        if isinstance(stmt, (ElseIf, Else)):
            raise NotImplementedError("Unexpected standalone else statement in C backend")
        raise NotImplementedError(f"C backend does not support statement {type(stmt).__name__}")

    def _render_assignment(self, stmt: Assignment) -> str:
        target = stmt.target
        if isinstance(target, (GetItem, GetAttr)):
            raise NotImplementedError("C backend does not support indexing or struct access")
        if not isinstance(target, Variable):
            raise NotImplementedError(
                f"C backend only supports assignment to variables, got {type(target).__name__}"
            )
        lhs = self._render_variable(target, as_pointer=False)
        rhs = self._render_expr(stmt.value)
        return f"{lhs} = {rhs};"

    def _render_if(self, stmt: If, indent: int) -> list[str]:
        lines: list[str] = []
        condition = self._render_expr(stmt.condition)
        lines.append(f"{'    ' * indent}if ({condition}) {{\n")
        lines.extend(self._render_statements(stmt.scope.get_statements(), indent + 1))
        for else_stmt in stmt.orelse:
            if isinstance(else_stmt, ElseIf):
                condition = self._render_expr(else_stmt.condition)
                lines.append(f"{'    ' * indent}}} else if ({condition}) {{\n")
                lines.extend(self._render_statements(else_stmt.scope.get_statements(), indent + 1))
            elif isinstance(else_stmt, Else):
                lines.append(f"{'    ' * indent}}} else {{\n")
                lines.extend(self._render_statements(else_stmt.scope.get_statements(), indent + 1))
            else:
                raise NotImplementedError(
                    f"C backend does not support else statement {type(else_stmt).__name__}"
                )
        lines.append(f"{'    ' * indent}}}\n")
        return lines

    def _render_do(self, stmt: Do, indent: int) -> list[str]:
        if not isinstance(stmt.iterator, Variable):
            raise NotImplementedError("C backend only supports variable iterators in loops")
        iterator = self._render_variable(stmt.iterator, as_pointer=False)
        start = self._render_expr(stmt.start)
        end = self._render_expr(stmt.end)
        step = "1" if stmt.step is None else self._render_expr(stmt.step)
        step_literal = None
        if isinstance(stmt.step, LiteralNode):
            step_literal = stmt.step.value
        condition = "<="
        if isinstance(step_literal, (int, float)) and step_literal < 0:
            condition = ">="
        init = f"{iterator} = {start}"
        cond = f"{iterator} {condition} {end}"
        incr = f"{iterator} += {step}" if step != "1" else f"{iterator}++"
        lines = [f"{'    ' * indent}for ({init}; {cond}; {incr}) {{\n"]
        lines.extend(self._render_statements(stmt.scope.get_statements(), indent + 1))
        lines.append(f"{'    ' * indent}}}\n")
        return lines

    def _render_call(self, stmt: Call) -> str:
        function = stmt.function
        if isinstance(function, str):
            name = function
            args = ", ".join(self._render_expr(arg) for arg in stmt.arguments)
            return f"{name}({args});"
        name = function.name
        arg_values: list[str] = []
        callee_args = list(function.arguments.values())
        if len(callee_args) != len(stmt.arguments):
            raise ValueError(
                f"C backend call mismatch for {name}: expected {len(callee_args)} args, "
                f"got {len(stmt.arguments)}"
            )
        for call_arg, callee_arg in zip(stmt.arguments, callee_args):
            callee_is_pointer = self._is_pointer_arg(callee_arg)
            arg_values.append(self._render_call_arg(call_arg, callee_is_pointer))
        args = ", ".join(arg_values)
        return f"{name}({args});"

    def _render_call_arg(self, arg, callee_is_pointer: bool) -> str:
        if callee_is_pointer:
            if isinstance(arg, Variable):
                if self._pointer_args.get(arg.name, False):
                    return arg.name
                return f"&{arg.name}"
            raise NotImplementedError("C backend requires variables for inout arguments")
        return self._render_expr(arg)

    def _render_expr(self, expr) -> str:
        if isinstance(expr, LiteralNode):
            return self._render_literal(expr.value)
        if isinstance(expr, Variable):
            return self._render_variable(expr, as_pointer=False)
        if isinstance(expr, BinaryOperationNode):
            if expr.op == "**":
                self._requires_math = True
                return f"pow({self._render_expr(expr.left)}, {self._render_expr(expr.right)})"
            op = _C_OPS.get(expr.op, expr.op)
            return f"({self._render_expr(expr.left)} {op} {self._render_expr(expr.right)})"
        if isinstance(expr, FunctionCall):
            name = expr.function.name
            args = ", ".join(self._render_expr(arg) for arg in expr.arguments)
            return f"{name}({args})"
        if isinstance(expr, IntrinsicFunction):
            return self._render_intrinsic(expr)
        if isinstance(expr, (GetItem, GetAttr)):
            raise NotImplementedError("C backend does not support indexing or struct access")
        raise NotImplementedError(f"C backend does not support expression {type(expr).__name__}")

    def _render_intrinsic(self, expr: IntrinsicFunction) -> str:
        token = expr.token
        if token == "-":
            return f"(-{self._render_expr(expr.arguments[0])})"
        if token == ".not.":
            return f"(!{self._render_expr(expr.arguments[0])})"
        if token in {"abs", "exp", "sqrt", "sin", "cos", "tan", "asin", "acos", "atan"}:
            self._requires_math = True
            args = ", ".join(self._render_expr(arg) for arg in expr.arguments)
            return f"{token}({args})"
        raise NotImplementedError(f"C backend does not support intrinsic {token}")

    def _render_literal(self, value) -> str:
        if isinstance(value, (bool, np.bool_)):
            return "1" if value else "0"
        if isinstance(value, (int, float, np.integer, np.floating)):
            return repr(value)
        raise NotImplementedError("C backend only supports int/float/bool literals")

    def _render_variable(self, variable: Variable, as_pointer: bool) -> str:
        if self._pointer_args.get(variable.name, False):
            return variable.name if as_pointer else f"*{variable.name}"
        return variable.name

    def _is_pointer_arg(self, variable: Variable) -> bool:
        if variable._shape is not SCALAR:
            return True
        dtype = DataType.from_ftype(variable._ftype)
        if variable.intent == "in" and dtype.can_be_value():
            return False
        return True

    @staticmethod
    def _validate_scalar(variable: Variable) -> None:
        if variable._shape is not SCALAR:
            raise NotImplementedError("C backend currently supports only scalar arguments")
