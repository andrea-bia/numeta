from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from numeta.array_shape import SCALAR, UNKNOWN
from numeta.datatype import DataType
from numeta.syntax import Subroutine, Variable
from numeta.syntax.settings import settings as syntax_settings
from numeta.settings import settings as nm_settings
from numeta.syntax.expressions import (
    BinaryOperationNode,
    FunctionCall,
    IntrinsicFunction,
    LiteralNode,
)
from numeta.syntax.expressions.getattr import GetAttr
from numeta.syntax.expressions.getitem import GetItem
from numeta.syntax.expressions.various import ArrayConstructor
from numeta.syntax.statements import (
    Allocate,
    Assignment,
    Call,
    Case,
    Cycle,
    Deallocate,
    Do,
    DoWhile,
    Else,
    ElseIf,
    Exit,
    If,
    PointerAssignment,
    Print,
    Return,
    SelectCase,
    Stop,
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


@dataclass
class ArrayInfo:
    name: str
    ctype: str
    rank: int
    fortran_order: bool
    dims_exprs: list[str]
    dims_name: str | None
    is_argument: bool
    is_pointer: bool


class CCodegen:
    def __init__(self, subroutine: Subroutine) -> None:
        self.subroutine = subroutine
        self._pointer_args: dict[str, bool] = {}
        self._arg_specs: list[CArg] = []
        self._array_info: dict[str, ArrayInfo] = {}
        self._struct_defs: list[str] = []
        self._requires_math = False
        self._loop_counter = 0
        self._build_signature()
        self._collect_structs()

    @property
    def requires_math(self) -> bool:
        return self._requires_math

    def render(self) -> str:
        lines: list[str] = []
        lines.append("#include <Python.h>\n")
        lines.append("#include <numpy/arrayobject.h>\n")
        if self._requires_math:
            lines.append("#include <math.h>\n")
        lines.append("\n")
        lines.extend(self._struct_defs)
        if self._struct_defs:
            lines.append("\n")
        prototypes = self._render_prototypes()
        lines.extend(prototypes)
        if prototypes:
            lines.append("\n")
        lines.extend(self._render_function())
        return "".join(lines)

    def _build_signature(self) -> None:
        for variable in self.subroutine.arguments.values():
            if variable._shape is SCALAR:
                ctype = self._map_ftype_to_ctype(variable)
                is_const = variable.intent == "in"
                is_pointer = self._is_pointer_arg(variable)
                self._pointer_args[variable.name] = is_pointer
                self._arg_specs.append(CArg(variable.name, ctype, is_pointer, is_const))
            else:
                self._register_array_argument(variable)

    def _collect_structs(self) -> None:
        defs: dict[str, str] = {}

        def bfs(dtype):
            if not dtype.is_struct():
                return
            if dtype.name in defs:
                return
            for _, nested_dtype, _ in dtype.members:
                bfs(nested_dtype)
            defs[dtype.name] = dtype.c_declaration()

        variables = list(self.subroutine.arguments.values())
        variables.extend(self.subroutine.get_local_variables().values())
        for variable in variables:
            dtype = self._safe_dtype_from_ftype(variable)
            if dtype is not None and dtype.is_struct():
                bfs(dtype)

        self._struct_defs = list(defs.values())

    def _register_array_argument(self, variable: Variable) -> None:
        ctype = self._map_ftype_to_ctype(variable)
        fortran_order = variable._shape.fortran_order
        dims_name = None
        dims_exprs: list[str] = []
        has_shape_descriptor = (
            nm_settings.add_shape_descriptors and variable._shape.has_comptime_undefined_dims()
        )
        if has_shape_descriptor:
            dims_name = f"{variable.name}_dims"
            self._arg_specs.append(CArg(dims_name, "npy_intp", True, True))
            dims_exprs = [f"{dims_name}[{i}]" for i in range(rank)]
        else:
            dims_exprs = [self._render_dim(dim) for dim in variable._shape.dims]

        self._arg_specs.append(CArg(variable.name, ctype, True, variable.intent == "in"))
        self._pointer_args[variable.name] = True
        self._array_info[variable.name] = ArrayInfo(
            name=variable.name,
            ctype=ctype,
            rank=rank,
            fortran_order=fortran_order,
            dims_exprs=dims_exprs,
            dims_name=dims_name,
            is_argument=True,
            is_pointer=True,
        )

    def _render_function(self) -> list[str]:
        args = ", ".join(arg.render() for arg in self._arg_specs)
        header = f"void {self.subroutine.name}({args}) {{\n"
        lines = [header]
        lines.extend(self._render_local_declarations(indent=1))
        lines.extend(self._render_statements(self.subroutine.scope.get_statements(), indent=1))
        lines.append("}\n")
        return lines

    def _render_prototypes(self) -> list[str]:
        prototypes = []
        for sub in self._collect_called_subroutines():
            if sub.name == self.subroutine.name:
                continue
            args = ", ".join(self._render_prototype_arg(sub, var) for var in sub.arguments.values())
            prototypes.append(f"void {sub.name}({args});\n")
        return prototypes

    def _collect_called_subroutines(self) -> list[Subroutine]:
        seen = set()
        collected = []

        def add_sub(sub):
            if sub.name in seen:
                return
            seen.add(sub.name)
            collected.append(sub)

        def walk_expr(expr):
            if isinstance(expr, FunctionCall):
                if isinstance(expr.function, Subroutine):
                    add_sub(expr.function)
                for arg in expr.arguments:
                    walk_expr(arg)
            elif isinstance(expr, BinaryOperationNode):
                walk_expr(expr.left)
                walk_expr(expr.right)
            elif isinstance(expr, IntrinsicFunction):
                for arg in expr.arguments:
                    walk_expr(arg)
            elif isinstance(expr, GetItem):
                walk_expr(expr.variable)
                if isinstance(expr.sliced, tuple):
                    for item in expr.sliced:
                        if hasattr(item, "get_code_blocks"):
                            walk_expr(item)
                elif hasattr(expr.sliced, "get_code_blocks"):
                    walk_expr(expr.sliced)

        def walk_stmt(stmt):
            if isinstance(stmt, Call) and isinstance(stmt.function, Subroutine):
                add_sub(stmt.function)
            if isinstance(stmt, Assignment):
                walk_expr(stmt.target)
                walk_expr(stmt.value)
            if isinstance(stmt, If):
                walk_expr(stmt.condition)
                for s in stmt.scope.get_statements():
                    walk_stmt(s)
                for o in stmt.orelse:
                    if isinstance(o, (ElseIf, Else)):
                        for s in o.scope.get_statements():
                            walk_stmt(s)
            if isinstance(stmt, Do):
                walk_expr(stmt.iterator)
                walk_expr(stmt.start)
                walk_expr(stmt.end)
                if stmt.step is not None:
                    walk_expr(stmt.step)
                for s in stmt.scope.get_statements():
                    walk_stmt(s)
            if isinstance(stmt, DoWhile):
                walk_expr(stmt.condition)
                for s in stmt.scope.get_statements():
                    walk_stmt(s)
            if isinstance(stmt, SelectCase):
                walk_expr(stmt.value)
                for s in stmt.scope.get_statements():
                    walk_stmt(s)
            if isinstance(stmt, Case):
                walk_expr(stmt.value)
                for s in stmt.scope.get_statements():
                    walk_stmt(s)
            if isinstance(stmt, Print):
                for child in stmt.children:
                    if hasattr(child, "get_code_blocks"):
                        walk_expr(child)

        for stmt in self.subroutine.scope.get_statements():
            walk_stmt(stmt)

        return collected

    def _render_prototype_arg(self, sub: Subroutine, variable: Variable) -> str:
        if variable._shape is SCALAR:
            ctype = self._map_ftype_to_ctype(variable)
            is_const = variable.intent == "in"
            is_pointer = self._is_pointer_arg(variable)
            return CArg(variable.name, ctype, is_pointer, is_const).render()
        rank = variable._shape.rank
        ctype = self._map_ftype_to_ctype(variable)
        args = []
        has_shape_descriptor = (
            nm_settings.add_shape_descriptors and variable._shape.has_comptime_undefined_dims()
        )
        if has_shape_descriptor:
            args.append(CArg(f"{variable.name}_dims", "npy_intp", True, True).render())
        args.append(CArg(variable.name, ctype, True, variable.intent == "in").render())
        return ", ".join(args)

    def _render_local_declarations(self, indent: int) -> list[str]:
        declarations: list[str] = []
        for variable in self.subroutine.get_local_variables().values():
            if variable._shape is SCALAR:
                ctype = self._map_ftype_to_ctype(variable)
                const_prefix = "const " if variable.parameter else ""
                init = ""
                if variable.assign is not None:
                    init_value = self._render_literal(variable.assign)
                    init = f" = {init_value}"
                declarations.append(
                    f"{'    ' * indent}{const_prefix}{ctype} {variable.name}{init};\n"
                )
                continue

            self._register_local_array(variable)
            info = self._array_info[variable.name]
            if info.dims_name is not None and not info.is_argument:
                dims_init = ""
                if not variable._shape.has_comptime_undefined_dims():
                    dims_vals = ", ".join(info.dims_exprs)
                    dims_init = f" = {{{dims_vals}}}"
                declarations.append(
                    f"{'    ' * indent}npy_intp {info.dims_name}[{info.rank}]{dims_init};\n"
                )

            if (
                variable.allocatable
                or variable.pointer
                or variable._shape.has_comptime_undefined_dims()
            ):
                declarations.append(f"{'    ' * indent}{info.ctype} *{variable.name} = NULL;\n")
                continue

            total = self._render_product(info.dims_exprs)
            declarations.append(f"{'    ' * indent}{info.ctype} {variable.name}[{total}];\n")

        if declarations:
            declarations.append("\n")
        return declarations

    def _register_local_array(self, variable: Variable) -> None:
        if variable.name in self._array_info:
            return
        rank = variable._shape.rank
        ctype = self._map_ftype_to_ctype(variable)
        fortran_order = variable._shape.fortran_order
        dims_name = f"{variable.name}_dims"
        if variable._shape is UNKNOWN or variable._shape.has_comptime_undefined_dims():
            dims_exprs = [f"{dims_name}[{i}]" for i in range(rank)]
        else:
            dims_exprs = [self._render_dim(dim) for dim in variable._shape.dims]

        self._array_info[variable.name] = ArrayInfo(
            name=variable.name,
            ctype=ctype,
            rank=rank,
            fortran_order=fortran_order,
            dims_exprs=dims_exprs,
            dims_name=dims_name,
            is_argument=False,
            is_pointer=True,
        )

    def _render_statements(self, statements: Iterable, indent: int) -> list[str]:
        lines: list[str] = []
        for stmt in statements:
            lines.extend(self._render_statement(stmt, indent=indent))
        return lines

    def _render_statement(self, stmt, indent: int) -> list[str]:
        if isinstance(stmt, Assignment):
            return self._render_assignment_statement(stmt, indent)
        if isinstance(stmt, If):
            return self._render_if(stmt, indent)
        if isinstance(stmt, Do):
            return self._render_do(stmt, indent)
        if isinstance(stmt, DoWhile):
            return self._render_do_while(stmt, indent)
        if isinstance(stmt, Return):
            return [f"{'    ' * indent}return;\n"]
        if isinstance(stmt, Call):
            return [f"{'    ' * indent}{self._render_call(stmt)}\n"]
        if isinstance(stmt, Cycle):
            return [f"{'    ' * indent}continue;\n"]
        if isinstance(stmt, Exit):
            return [f"{'    ' * indent}break;\n"]
        if isinstance(stmt, Stop):
            return [f"{'    ' * indent}return;\n"]
        if isinstance(stmt, Print):
            return self._render_print(stmt, indent)
        if isinstance(stmt, SelectCase):
            return self._render_select_case(stmt, indent)
        if isinstance(stmt, Allocate):
            return self._render_allocate(stmt, indent)
        if isinstance(stmt, Deallocate):
            return self._render_deallocate(stmt, indent)
        if isinstance(stmt, PointerAssignment):
            return self._render_pointer_assignment(stmt, indent)
        if isinstance(stmt, (ElseIf, Else, Case)):
            raise NotImplementedError("Unexpected standalone block statement in C backend")
        raise NotImplementedError(f"C backend does not support statement {type(stmt).__name__}")

    def _render_assignment_statement(self, stmt: Assignment, indent: int) -> list[str]:
        target = stmt.target
        if self._is_array_target(target):
            return self._render_array_assignment(stmt, indent)
        if isinstance(target, GetAttr):
            lhs = self._render_getattr(target)
            rhs = self._render_expr(stmt.value)
            return [f"{'    ' * indent}{lhs} = {rhs};\n"]
        if not isinstance(target, Variable):
            raise NotImplementedError(
                f"C backend only supports assignment to variables or array targets, got {type(target).__name__}"
            )
        lhs = self._render_variable(target, as_pointer=False)
        rhs = self._render_expr(stmt.value)
        return [f"{'    ' * indent}{lhs} = {rhs};\n"]

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

    def _render_do_while(self, stmt: DoWhile, indent: int) -> list[str]:
        condition = self._render_expr(stmt.condition)
        lines = [f"{'    ' * indent}while ({condition}) {{\n"]
        lines.extend(self._render_statements(stmt.scope.get_statements(), indent + 1))
        lines.append(f"{'    ' * indent}}}\n")
        return lines

    def _render_select_case(self, stmt: SelectCase, indent: int) -> list[str]:
        value = self._render_expr(stmt.value)
        lines = [f"{'    ' * indent}switch ({value}) {{\n"]
        for case_stmt in stmt.scope.get_statements():
            if not isinstance(case_stmt, Case):
                raise NotImplementedError("C backend only supports case blocks in select case")
            case_value = self._render_expr(case_stmt.value)
            lines.append(f"{'    ' * (indent + 1)}case {case_value}:\n")
            lines.extend(self._render_statements(case_stmt.scope.get_statements(), indent + 2))
            lines.append(f"{'    ' * (indent + 2)}break;\n")
        lines.append(f"{'    ' * indent}}}\n")
        return lines

    def _render_allocate(self, stmt: Allocate, indent: int) -> list[str]:
        if not isinstance(stmt.target, Variable):
            raise NotImplementedError("C backend only supports allocate on variables")
        if stmt.target.name not in self._array_info:
            self._register_local_array(stmt.target)
        info = self._array_info[stmt.target.name]
        dims = [self._render_expr(dim) for dim in stmt.shape]
        if not info.fortran_order:
            dims = dims[::-1]
        size = self._render_product(dims)
        lines = []
        if info.dims_name is not None:
            for i, dim in enumerate(dims):
                lines.append(f"{'    ' * indent}{info.dims_name}[{i}] = {dim};\n")
        lines.append(
            f"{'    ' * indent}{stmt.target.name} = ({info.ctype}*)PyDataMem_NEW(sizeof({info.ctype}) * {size});\n"
        )
        return lines

    def _render_deallocate(self, stmt: Deallocate, indent: int) -> list[str]:
        if not isinstance(stmt.array, Variable):
            raise NotImplementedError("C backend only supports deallocate on variables")
        return [f"{'    ' * indent}PyDataMem_FREE({stmt.array.name});\n"]

    def _render_pointer_assignment(self, stmt: PointerAssignment, indent: int) -> list[str]:
        if not isinstance(stmt.pointer, Variable):
            raise NotImplementedError("C backend only supports pointer assignment to variables")
        target_expr = self._render_pointer_target(stmt.target, stmt.pointer_shape)
        return [f"{'    ' * indent}{stmt.pointer.name} = {target_expr};\n"]

    def _render_call(self, stmt: Call) -> str:
        function = stmt.function
        name = function if isinstance(function, str) else function.name
        arg_values: list[str] = []
        if isinstance(function, str):
            args = ", ".join(self._render_expr(arg) for arg in stmt.arguments)
            return f"{name}({args});"

        callee_args = list(function.arguments.values())
        if len(callee_args) != len(stmt.arguments):
            raise ValueError(
                f"C backend call mismatch for {name}: expected {len(callee_args)} args, "
                f"got {len(stmt.arguments)}"
            )
        for call_arg, callee_arg in zip(stmt.arguments, callee_args):
            arg_values.extend(self._render_call_arg(call_arg, callee_arg))
        args = ", ".join(arg_values)
        return f"{name}({args});"

    def _render_call_arg(self, arg, callee_arg: Variable) -> list[str]:
        if callee_arg._shape is SCALAR:
            callee_is_pointer = self._is_pointer_arg(callee_arg)
            if callee_is_pointer:
                if isinstance(arg, Variable):
                    if self._pointer_args.get(arg.name, False):
                        return [arg.name]
                    return [f"&{arg.name}"]
                raise NotImplementedError("C backend requires variables for inout arguments")
            return [self._render_expr(arg)]

        if isinstance(arg, Variable):
            info = self._array_info.get(arg.name)
            if info is None:
                self._register_local_array(arg)
                info = self._array_info[arg.name]
            args: list[str] = []
            if (
                callee_arg._shape.has_comptime_undefined_dims()
                and nm_settings.add_shape_descriptors
            ):
                if info.dims_name is not None:
                    args.append(info.dims_name)
                else:
                    dims_tmp = f"{arg.name}_dims"
                    args.append(dims_tmp)
            args.append(arg.name)
            return args
        raise NotImplementedError("C backend only supports array arguments as variables")

    def _render_print(self, stmt: Print, indent: int) -> list[str]:
        parts: list[str] = []
        for child in stmt.children:
            if isinstance(child, str):
                parts.append(f'"{child}"')
            else:
                fmt = self._format_for_expr(child)
                parts.append(f"{fmt}, {self._render_expr(child)}")
        if not parts:
            return []
        lines = []
        for part in parts:
            lines.append(f"{'    ' * indent}printf({part});\n")
            lines.append(f"{'    ' * indent}printf(\" \");\n")
        lines.append(f"{'    ' * indent}printf(\"\\n\");\n")
        return lines

    def _render_array_assignment(self, stmt: Assignment, indent: int) -> list[str]:
        target = stmt.target
        target_shape = target._shape
        if target_shape is UNKNOWN:
            raise NotImplementedError("C backend does not support assignments to unknown shapes")
        loop_dims, target_indices = self._build_target_indices(target)
        rhs_is_scalar = stmt.value._shape is SCALAR
        if not rhs_is_scalar and stmt.value._shape.rank != target_shape.rank:
            raise NotImplementedError("C backend requires matching shapes for array assignment")
        lines: list[str] = []
        loop_vars: list[str] = []
        for dim in loop_dims:
            var = self._next_loop_var()
            loop_vars.append(var)
            start, end, step = dim
            step_expr = step or "1"
            condition = "<="
            step_literal = None
            if isinstance(step, str):
                try:
                    step_literal = int(step)
                except ValueError:
                    step_literal = None
            if isinstance(step_literal, int) and step_literal < 0:
                condition = ">="
            lines.append(
                f"{'    ' * indent}for (npy_intp {var} = {start}; {var} {condition} {end}; {var} += {step_expr}) {{\n"
            )
            indent += 1

        element_indices = []
        loop_iter = iter(loop_vars)
        for idx in target_indices:
            if idx is None:
                element_indices.append(next(loop_iter))
            else:
                element_indices.append(idx)
        target_var = target.variable if isinstance(target, GetItem) else target
        if not isinstance(target_var, Variable):
            raise NotImplementedError("C backend only supports variable array targets")
        lhs = self._render_array_element(target_var, element_indices)
        rhs = (
            self._render_expr_at(stmt.value, element_indices)
            if not rhs_is_scalar
            else self._render_expr(stmt.value)
        )
        lines.append(f"{'    ' * indent}{lhs} = {rhs};\n")

        for _ in loop_dims:
            indent -= 1
            lines.append(f"{'    ' * indent}}}\n")
        return lines

    def _build_target_indices(
        self, target
    ) -> tuple[list[tuple[str, str, str | None]], list[str | None]]:
        if isinstance(target, Variable):
            info = self._array_info[target.name]
            loop_dims = [
                (
                    str(syntax_settings.array_lower_bound),
                    f"{info.dims_exprs[i]} - 1 + {syntax_settings.array_lower_bound}",
                    None,
                )
                for i in range(info.rank)
            ]
            return loop_dims, [None] * info.rank
        if not isinstance(target, GetItem):
            raise NotImplementedError(
                "C backend only supports array targets as variables or getitem"
            )
        variable = target.variable
        info = self._array_info[variable.name]
        slices = target.sliced if isinstance(target.sliced, tuple) else (target.sliced,)
        slices = list(slices)
        if len(slices) < info.rank:
            slices.extend([slice(None)] * (info.rank - len(slices)))
        loop_dims: list[tuple[str, str, str | None]] = []
        target_indices: list[str | None] = []
        for i, slice_ in enumerate(slices):
            if isinstance(slice_, slice):
                if slice_.step is not None:
                    step = self._render_expr(slice_.step)
                else:
                    step = None
                start, end = self._slice_bounds(slice_, info.dims_exprs[i])
                loop_dims.append((start, end, step))
                target_indices.append(None)
            else:
                idx = self._render_expr(slice_)
                target_indices.append(idx)
        return loop_dims, target_indices

    def _render_expr_at(self, expr, indices: Sequence[str]) -> str:
        if isinstance(expr, LiteralNode):
            return self._render_literal(expr.value)
        if isinstance(expr, Variable):
            if expr._shape is SCALAR:
                return self._render_variable(expr, as_pointer=False)
            return self._render_array_element(expr, indices)
        if isinstance(expr, GetItem):
            return self._render_getitem(expr, indices)
        if isinstance(expr, BinaryOperationNode):
            if expr.op == "**":
                self._requires_math = True
                return f"pow({self._render_expr_at(expr.left, indices)}, {self._render_expr_at(expr.right, indices)})"
            op = _C_OPS.get(expr.op, expr.op)
            return f"({self._render_expr_at(expr.left, indices)} {op} {self._render_expr_at(expr.right, indices)})"
        if isinstance(expr, FunctionCall):
            name = expr.function.name
            args = ", ".join(self._render_expr(arg) for arg in expr.arguments)
            return f"{name}({args})"
        if isinstance(expr, IntrinsicFunction):
            return self._render_intrinsic(expr, indices)
        if isinstance(expr, GetAttr):
            return self._render_getattr(expr)
        raise NotImplementedError(f"C backend does not support expression {type(expr).__name__}")

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
        if isinstance(expr, GetItem):
            return self._render_getitem(expr)
        if isinstance(expr, GetAttr):
            return self._render_getattr(expr)
        if isinstance(expr, ArrayConstructor):
            raise NotImplementedError(
                "C backend does not support array constructors in expressions"
            )
        raise NotImplementedError(f"C backend does not support expression {type(expr).__name__}")

    def _render_intrinsic(
        self, expr: IntrinsicFunction, indices: Sequence[str] | None = None
    ) -> str:
        token = expr.token
        if token == "-":
            arg = (
                self._render_expr_at(expr.arguments[0], indices)
                if indices
                else self._render_expr(expr.arguments[0])
            )
            return f"(-{arg})"
        if token == ".not.":
            arg = (
                self._render_expr_at(expr.arguments[0], indices)
                if indices
                else self._render_expr(expr.arguments[0])
            )
            return f"(!{arg})"
        if token in {"abs", "exp", "sqrt", "sin", "cos", "tan", "asin", "acos", "atan"}:
            self._requires_math = True
            args = ", ".join(
                self._render_expr_at(arg, indices) if indices else self._render_expr(arg)
                for arg in expr.arguments
            )
            return f"{token}({args})"
        raise NotImplementedError(f"C backend does not support intrinsic {token}")

    def _render_getitem(self, expr: GetItem, indices: Sequence[str] | None = None) -> str:
        variable = expr.variable
        if not isinstance(variable, Variable):
            raise NotImplementedError("C backend only supports getitem on variables")
        info = self._array_info.get(variable.name)
        if info is None:
            raise NotImplementedError("C backend requires array metadata for getitem")
        slices = expr.sliced if isinstance(expr.sliced, tuple) else (expr.sliced,)
        slices = list(slices)
        if len(slices) < info.rank:
            slices.extend([slice(None)] * (info.rank - len(slices)))
        out_indices: list[str] = []
        index_iter = iter(indices) if indices is not None else None
        for i, slice_ in enumerate(slices):
            if isinstance(slice_, slice):
                if index_iter is None:
                    raise NotImplementedError("C backend requires indices for sliced getitem")
                start, _ = self._slice_bounds(slice_, info.dims_exprs[i])
                out_indices.append(f"({start} + {next(index_iter)})")
            else:
                idx = self._render_expr(slice_)
                out_indices.append(idx)
        return self._render_array_element(variable, out_indices)

    def _render_getattr(self, expr: GetAttr) -> str:
        base = expr.variable
        if isinstance(base, Variable):
            if self._pointer_args.get(base.name, False):
                return f"{base.name}->{expr.attr}"
            return f"{base.name}.{expr.attr}"
        raise NotImplementedError("C backend only supports getattr on variables")

    def _render_array_element(self, variable: Variable, indices: Sequence[str]) -> str:
        info = self._array_info.get(variable.name)
        if info is None:
            raise NotImplementedError("C backend requires array metadata for element access")
        if len(indices) != info.rank:
            raise ValueError("Index rank does not match array rank")
        linear = self._linear_index(indices, info.dims_exprs, info.fortran_order)
        return f"{variable.name}[{linear}]"

    def _linear_index(
        self, indices: Sequence[str], dims: Sequence[str], fortran_order: bool
    ) -> str:
        if len(indices) == 1:
            return self._normalize_index(indices[0])
        if fortran_order:
            stride = "1"
            terms = []
            for i, idx in enumerate(indices):
                term = f"({self._normalize_index(idx)})"
                if stride != "1":
                    term = f"({term})*({stride})"
                terms.append(term)
                stride = f"({stride})*({dims[i]})"
            return " + ".join(terms)
        stride = "1"
        terms = []
        for i in range(len(indices) - 1, -1, -1):
            term = f"({self._normalize_index(indices[i])})"
            if stride != "1":
                term = f"({term})*({stride})"
            terms.append(term)
            stride = f"({stride})*({dims[i]})"
        return " + ".join(reversed(terms))

    def _slice_bounds(self, slice_: slice, dim_expr: str) -> tuple[str, str]:
        lb = syntax_settings.array_lower_bound
        start = slice_.start
        stop = slice_.stop
        if start is None:
            start_expr = str(lb)
        else:
            start_expr = self._render_expr(start)
        if stop is None:
            stop_expr = f"({dim_expr} - 1 + {lb})"
        else:
            stop_expr = self._render_expr(stop)
            if syntax_settings.c_like_bounds:
                stop_expr = f"({stop_expr} - 1)"
        return start_expr, stop_expr

    def _render_literal(self, value) -> str:
        if isinstance(value, (bool, np.bool_)):
            return "1" if value else "0"
        if isinstance(value, (int, float, np.integer, np.floating)):
            return repr(value)
        if isinstance(value, str):
            return f'"{value}"'
        raise NotImplementedError("C backend only supports int/float/bool/string literals")

    def _render_variable(self, variable: Variable, as_pointer: bool) -> str:
        if self._pointer_args.get(variable.name, False):
            return variable.name if as_pointer else f"*{variable.name}"
        return variable.name

    def _is_pointer_arg(self, variable: Variable) -> bool:
        if variable._shape is not SCALAR:
            return True
        dtype = self._safe_dtype_from_ftype(variable)
        if dtype is not None and variable.intent == "in" and dtype.can_be_value():
            return False
        return True

    def _map_ftype_to_ctype(self, variable: Variable) -> str:
        ftype = variable._ftype
        if ftype is None:
            raise NotImplementedError("C backend requires a concrete type")
        if ftype.type == "type" and getattr(ftype.kind, "name", None) == "c_ptr":
            return "void"
        dtype = DataType.from_ftype(ftype)
        return dtype.get_cnumpy()

    def _safe_dtype_from_ftype(self, variable: Variable) -> DataType | None:
        try:
            return DataType.from_ftype(variable._ftype)
        except Exception:
            return None

    def _format_for_expr(self, expr) -> str:
        if isinstance(expr, LiteralNode):
            if isinstance(expr.value, int):
                return '"%ld"'
            if isinstance(expr.value, float):
                return '"%g"'
            if isinstance(expr.value, str):
                return '"%s"'
            if isinstance(expr.value, bool):
                return '"%d"'
        if hasattr(expr, "_ftype"):
            ftype = expr._ftype
            if ftype is not None and ftype.type == "integer":
                return '"%ld"'
            if ftype is not None and ftype.type == "real":
                return '"%g"'
            if ftype is not None and ftype.type == "logical":
                return '"%d"'
        return '"%g"'

    def _render_dim(self, dim) -> str:
        if isinstance(dim, (int, np.integer)):
            return str(dim)
        return self._render_expr(dim)

    def _render_product(self, terms: Sequence[str]) -> str:
        if not terms:
            return "1"
        result = terms[0]
        for term in terms[1:]:
            result = f"({result})*({term})"
        return result

    def _normalize_index(self, idx: str) -> str:
        lb = syntax_settings.array_lower_bound
        if lb == 0:
            return idx
        return f"({idx} - {lb})"

    def _is_array_target(self, target) -> bool:
        if isinstance(target, Variable):
            return target._shape is not SCALAR
        if isinstance(target, GetItem):
            return True
        return False

    def _next_loop_var(self) -> str:
        name = f"_nm_i{self._loop_counter}"
        self._loop_counter += 1
        return name

    def _render_pointer_target(self, target, pointer_shape) -> str:
        if isinstance(target, Variable):
            if target._shape is SCALAR:
                return f"&{target.name}"
            return target.name
        if isinstance(target, GetItem):
            indices = []
            slices = target.sliced if isinstance(target.sliced, tuple) else (target.sliced,)
            slices = list(slices)
            info = self._array_info[target.variable.name]
            if len(slices) < info.rank:
                slices.extend([slice(None)] * (info.rank - len(slices)))
            for i, slice_ in enumerate(slices):
                if isinstance(slice_, slice):
                    start, _ = self._slice_bounds(slice_, info.dims_exprs[i])
                    indices.append(start)
                else:
                    indices.append(self._render_expr(slice_))
            if not isinstance(target.variable, Variable):
                raise NotImplementedError("C backend only supports pointer assignment to variables")
            return f"&{self._render_array_element(target.variable, indices)}"
        raise NotImplementedError(
            "C backend only supports pointer assignment to variables or getitem"
        )
