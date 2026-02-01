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
from numeta.syntax.expressions.various import ArrayConstructor, Re, Im
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
        self._temp_counter = 0
        self._build_signature()
        self._collect_structs()

    @property
    def requires_math(self) -> bool:
        return self._requires_math

    def render(self) -> str:
        lines: list[str] = []
        lines.append("#include <Python.h>\n")
        lines.append("#include <numpy/arrayobject.h>\n")
        lines.append("#include <numpy/npy_math.h>\n")
        lines.append("#include <complex.h>\n")
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
        rank = variable._shape.rank
        ctype = self._map_ftype_to_ctype(variable)
        fortran_order = variable._shape.fortran_order
        dims_name = None
        dims_exprs: list[str] = []
        has_shape_descriptor = (
            nm_settings.add_shape_descriptors and variable._shape.has_comptime_undefined_dims()
        )
        has_shape_exprs = any(hasattr(dim, "get_code_blocks") for dim in variable._shape.dims)
        if has_shape_descriptor and not has_shape_exprs:
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
        if isinstance(target, (Re, Im)):
            line = self._render_complex_set(target, stmt.value)
            return [f"{'    ' * indent}{line}\n"]
        if isinstance(target, GetAttr):
            lhs = self._render_getattr(target)
            rhs, setup = self._render_expr_for_target(stmt.value, [], [], indent)
            lines = setup
            lines.append(f"{'    ' * indent}{lhs} = {rhs};\n")
            return lines
        if not isinstance(target, Variable):
            raise NotImplementedError(
                f"C backend only supports assignment to variables or array targets, got {type(target).__name__}"
            )
        lhs = self._render_variable(target, as_pointer=False)
        rhs, setup = self._render_expr_for_target(stmt.value, [], [], indent)
        lines = setup
        lines.append(f"{'    ' * indent}{lhs} = {rhs};\n")
        return lines

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
                if self._is_complex_expr(child):
                    rendered = self._render_expr(child)
                    suffix = "f" if self._is_complex64(child) else ""
                    parts.append(
                        f'"(%g%+gi)", creal{suffix}({rendered}), cimag{suffix}({rendered})'
                    )
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
        target_dim_exprs = self._target_dim_exprs(target, loop_dims)
        if not rhs_is_scalar:
            self._validate_broadcast(stmt.value, target_dim_exprs)
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
        lhs = self._render_array_element_from_expr(target, element_indices)
        rhs, setup_lines = self._render_expr_for_target(
            stmt.value, element_indices, target_dim_exprs, indent
        )
        lines.extend(setup_lines)
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
        if isinstance(target, GetAttr):
            rank, dims_exprs, _, _ = self._array_expr_info(target)
            loop_dims = [
                (
                    str(syntax_settings.array_lower_bound),
                    f"{dims_exprs[i]} - 1 + {syntax_settings.array_lower_bound}",
                    None,
                )
                for i in range(rank)
            ]
            return loop_dims, [None] * rank
        if not isinstance(target, GetItem):
            raise NotImplementedError(
                "C backend only supports array targets as variables or getitem"
            )
        variable = target.variable
        rank, dims_exprs, _, _ = self._array_expr_info(variable)
        slices = target.sliced if isinstance(target.sliced, tuple) else (target.sliced,)
        slices = list(slices)
        if len(slices) < rank:
            slices.extend([slice(None)] * (rank - len(slices)))
        loop_dims: list[tuple[str, str, str | None]] = []
        target_indices: list[str | None] = []
        for i, slice_ in enumerate(slices):
            if isinstance(slice_, slice):
                if slice_.step is not None:
                    step = self._render_expr(slice_.step)
                else:
                    step = None
                start, end = self._slice_bounds(slice_, dims_exprs[i])
                loop_dims.append((start, end, step))
                target_indices.append(None)
            else:
                idx = self._render_expr(slice_)
                target_indices.append(idx)
        return loop_dims, target_indices

    def _target_dim_exprs(self, target, loop_dims: list[tuple[str, str, str | None]]) -> list[str]:
        if isinstance(target, Variable):
            info = self._array_info[target.name]
            return list(info.dims_exprs)
        if isinstance(target, GetAttr):
            _, dims_exprs, _, _ = self._array_expr_info(target)
            return list(dims_exprs)
        dims = []
        for start, end, step in loop_dims:
            if step is None or step == "1":
                dims.append(f"({end} - {start} + 1)")
            else:
                dims.append(f"(({end} - {start})/({step}) + 1)")
        return dims

    def _validate_broadcast(self, expr, target_dims: Sequence[str]) -> None:
        expr_rank, expr_dims = self._expr_rank_dims(expr)
        target_rank = len(target_dims)
        if expr_rank > target_rank:
            raise NotImplementedError("C backend does not support broadcasting to lower rank")
        if expr_rank == 0:
            return
        pad = target_rank - expr_rank
        for i, expr_dim in enumerate(expr_dims):
            target_dim = target_dims[pad + i]
            expr_literal = self._literal_dim(expr_dim)
            target_literal = self._literal_dim(target_dim)
            if expr_literal is None or target_literal is None:
                continue
            if expr_literal != 1 and expr_literal != target_literal:
                raise NotImplementedError("C backend only supports broadcastable shapes")

    def _expr_rank_dims(self, expr) -> tuple[int, list[str]]:
        shape = expr._shape
        if shape is SCALAR:
            return 0, []
        if shape is UNKNOWN:
            raise NotImplementedError("C backend does not support unknown shapes in broadcasting")
        if isinstance(expr, ArrayConstructor):
            return 1, [str(len(expr.elements))]
        if isinstance(expr, Variable):
            info = self._array_info.get(expr.name)
            if info is not None:
                return info.rank, list(info.dims_exprs)
        if isinstance(expr, GetAttr):
            rank, dims_exprs, _, _ = self._array_expr_info(expr)
            return rank, list(dims_exprs)
        dims = []
        for dim in shape.dims:
            dims.append(self._render_dim(dim))
        return shape.rank, dims

    def _literal_dim(self, dim_expr: str) -> int | None:
        if isinstance(dim_expr, int):
            return dim_expr
        if not isinstance(dim_expr, str):
            return None
        text = dim_expr.strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
        return None

    def _render_expr_for_target(
        self,
        expr,
        target_indices: Sequence[str],
        target_dims: Sequence[str],
        indent: int,
    ) -> tuple[str, list[str]]:
        if isinstance(expr, IntrinsicFunction):
            return self._render_intrinsic_for_target(expr, target_indices, target_dims, indent)
        if isinstance(expr, ArrayConstructor):
            return self._render_array_constructor_expr(expr, target_indices, target_dims, indent)
        if isinstance(expr, BinaryOperationNode):
            left, left_setup = self._render_expr_for_target(
                expr.left, target_indices, target_dims, indent
            )
            right, right_setup = self._render_expr_for_target(
                expr.right, target_indices, target_dims, indent
            )
            setup = left_setup + right_setup
            if self._is_complex_expr(expr.left) or self._is_complex_expr(expr.right):
                return self._render_complex_binop_from_strings(expr, left, right), setup
            if expr.op == "**":
                self._requires_math = True
                return f"pow({left}, {right})", setup
            op = _C_OPS.get(expr.op, expr.op)
            return f"({left} {op} {right})", setup
        if expr._shape is SCALAR:
            return self._render_expr(expr), []
        if isinstance(expr, (Variable, GetItem, GetAttr, Re, Im)):
            expr_indices = self._broadcast_indices(expr, target_indices, target_dims)
            return self._render_expr_at(expr, expr_indices), []
        raise NotImplementedError(
            f"C backend does not support array expression {type(expr).__name__}"
        )

    def _broadcast_indices(
        self, expr, target_indices: Sequence[str], target_dims: Sequence[str]
    ) -> list[str]:
        expr_rank, expr_dims = self._expr_rank_dims(expr)
        target_rank = len(target_indices)
        if expr_rank == 0:
            return []
        if expr_rank > target_rank:
            raise NotImplementedError("C backend does not support broadcasting to lower rank")
        pad = target_rank - expr_rank
        lb = str(syntax_settings.array_lower_bound)
        mapped: list[str] = []
        for i, expr_dim in enumerate(expr_dims):
            target_index = target_indices[pad + i]
            expr_literal = self._literal_dim(expr_dim)
            target_literal = self._literal_dim(target_dims[pad + i])
            if expr_literal == 1:
                mapped.append(lb)
            elif expr_literal is None or target_literal is None:
                mapped.append(target_index)
            elif expr_literal == target_literal:
                mapped.append(target_index)
            else:
                raise NotImplementedError("C backend only supports broadcastable shapes")
        return mapped

    def _render_array_constructor_expr(
        self,
        expr: ArrayConstructor,
        target_indices: Sequence[str],
        target_dims: Sequence[str],
        indent: int,
    ) -> tuple[str, list[str]]:
        expr_rank, expr_dims = self._expr_rank_dims(expr)
        if expr_rank != 1:
            raise NotImplementedError("C backend only supports 1-D array constructors")
        ctype = self._ctype_from_expr(expr.elements[0])
        temp_name = f"_nm_ctor{self._temp_counter}"
        self._temp_counter += 1
        elements = ", ".join(self._render_expr(el) for el in expr.elements)
        decl = f"{'    ' * indent}{ctype} {temp_name}[{expr_dims[0]}] = {{{elements}}};\n"
        expr_indices = self._broadcast_indices(expr, target_indices, target_dims)
        index = self._normalize_index(expr_indices[0])
        return f"{temp_name}[{index}]", [decl]

    def _render_intrinsic_for_target(
        self,
        expr: IntrinsicFunction,
        target_indices: Sequence[str],
        target_dims: Sequence[str],
        indent: int,
    ) -> tuple[str, list[str]]:
        token = expr.token
        if token in {"sum", "maxval", "minval", "all", "dot_product"}:
            return self._render_reduction(expr, indent)
        if token == "matmul":
            return self._render_matmul(expr, target_indices, target_dims, indent)
        if token == "transpose":
            return self._render_transpose(expr, target_indices)
        if token == "shape":
            return self._render_shape_element(expr, target_indices[0])
        if token in {"size", "rank"}:
            return self._render_scalar_intrinsic(expr), []
        if token in {"max", "min"}:
            return self._render_max_min(expr, target_indices, target_dims, indent)
        setups: list[str] = []
        args_rendered: list[str] = []
        for arg in expr.arguments:
            rendered, setup = self._render_expr_for_target(arg, target_indices, target_dims, indent)
            setups.extend(setup)
            args_rendered.append(rendered)
        if token == "-":
            return f"(-{args_rendered[0]})", setups
        if token == ".not.":
            return f"(!{args_rendered[0]})", setups
        if token in {
            "abs",
            "exp",
            "sqrt",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "atan2",
            "floor",
            "sinh",
            "cosh",
            "tanh",
        }:
            if self._is_complex_expr(expr.arguments[0]) and token not in {"atan2", "floor"}:
                return (
                    self._render_complex_intrinsic(token, args_rendered[0], expr.arguments[0]),
                    setups,
                )
            self._requires_math = True
            return f"{token}({', '.join(args_rendered)})", setups
        if token in {"real", "aimag"}:
            arg0 = expr.arguments[0]
            if self._is_complex_expr(arg0):
                suffix = "f" if self._is_complex64(arg0) else ""
                func = "creal" if token == "real" else "cimag"
                return f"{func}{suffix}({args_rendered[0]})", setups
            return args_rendered[0], setups
        if token == "conjg":
            arg0 = expr.arguments[0]
            if self._is_complex_expr(arg0):
                suffix = "f" if self._is_complex64(arg0) else ""
                self._requires_math = True
                return f"conj{suffix}({args_rendered[0]})", setups
            return args_rendered[0], setups
        if token == "cmplx":
            real = args_rendered[0]
            imag = args_rendered[1] if len(args_rendered) > 1 else "0.0"
            is_float = self._cmplx_is_float(expr)
            self._requires_math = True
            if is_float:
                return f"npy_cpackf((float){real}, (float){imag})", setups
            return f"npy_cpack((double){real}, (double){imag})", setups
        if token in {"iand", "ior", "xor"}:
            op = "&" if token == "iand" else ("|" if token == "ior" else "^")
            return f"({args_rendered[0]} {op} {args_rendered[1]})", setups
        if token == "ishft":
            return (
                f"(({args_rendered[1]}) >= 0 ? (({args_rendered[0]}) << ({args_rendered[1]})) : (({args_rendered[0]}) >> (-({args_rendered[1]}))))",
                setups,
            )
        if token == "ibset":
            return f"(({args_rendered[0]}) | (1ULL << ({args_rendered[1]})))", setups
        if token == "ibclr":
            return f"(({args_rendered[0]}) & ~(1ULL << ({args_rendered[1]})))", setups
        if token == "popcnt":
            return f"__builtin_popcountll((unsigned long long)({args_rendered[0]}))", setups
        if token == "trailz":
            width = self._bit_width(expr.arguments[0])
            return (
                f"(({args_rendered[0]}) == 0 ? {width} : __builtin_ctzll((unsigned long long)({args_rendered[0]})))",
                setups,
            )
        if token == "allocated":
            arg0 = expr.arguments[0]
            if isinstance(arg0, Variable) and arg0._shape is not SCALAR:
                return f"({arg0.name} != NULL)", setups
            return "1", setups
        raise NotImplementedError(f"C backend does not support intrinsic {token}")

    def _array_expr_info(self, expr) -> tuple[int, list[str], bool, str]:
        if isinstance(expr, Variable):
            info = self._array_info.get(expr.name)
            if info is None:
                raise NotImplementedError("C backend requires array metadata for variable arrays")
            return info.rank, list(info.dims_exprs), info.fortran_order, expr.name
        if isinstance(expr, GetAttr):
            shape = expr._shape
            if shape is UNKNOWN or shape is SCALAR:
                raise NotImplementedError("C backend requires array-shaped GetAttr")
            dims_exprs = [self._render_dim(dim) for dim in shape.dims]
            return shape.rank, dims_exprs, shape.fortran_order, self._render_getattr(expr)
        raise NotImplementedError(
            "C backend only supports array metadata for variables or struct fields"
        )

    def _render_array_element_from_expr(self, expr, indices: Sequence[str]) -> str:
        if isinstance(expr, GetItem):
            return self._render_getitem(expr, indices)
        rank, dims_exprs, fortran_order, base = self._array_expr_info(expr)
        if len(indices) != rank:
            raise ValueError("Index rank does not match array rank")
        linear = self._linear_index(indices, dims_exprs, fortran_order)
        return f"{base}[{linear}]"

    def _render_complex_binop_from_strings(
        self, expr: BinaryOperationNode, left: str, right: str
    ) -> str:
        self._requires_math = True
        if expr.op == "**":
            func = (
                "cpowf"
                if (self._is_complex64(expr.left) or self._is_complex64(expr.right))
                else "cpow"
            )
            return f"{func}({left}, {right})"
        op = _C_OPS.get(expr.op, expr.op)
        return f"({left} {op} {right})"

    def _render_expr_at(self, expr, indices: Sequence[str]) -> str:
        if isinstance(expr, LiteralNode):
            return self._render_literal(expr.value)
        if isinstance(expr, Variable):
            if expr._shape is SCALAR:
                return self._render_variable(expr, as_pointer=False)
            return self._render_array_element(expr, indices)
        if isinstance(expr, GetItem):
            return self._render_getitem(expr, indices)
        if isinstance(expr, Re):
            return self._render_complex_component(expr.variable, "real", indices)
        if isinstance(expr, Im):
            return self._render_complex_component(expr.variable, "imag", indices)
        if isinstance(expr, GetAttr):
            if expr._shape is SCALAR:
                return self._render_getattr(expr)
            return self._render_array_element_from_expr(expr, indices)
        if isinstance(expr, BinaryOperationNode):
            if self._is_complex_expr(expr.left) or self._is_complex_expr(expr.right):
                return self._render_complex_binop(expr, indices)
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
        if isinstance(expr, (int, float, complex, bool, np.generic)):
            return self._render_literal(expr)
        if isinstance(expr, LiteralNode):
            return self._render_literal(expr.value)
        if isinstance(expr, Variable):
            return self._render_variable(expr, as_pointer=False)
        if isinstance(expr, BinaryOperationNode):
            if self._is_complex_expr(expr.left) or self._is_complex_expr(expr.right):
                return self._render_complex_binop(expr, None)
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
        if isinstance(expr, Re):
            return self._render_complex_component(expr.variable, "real", None)
        if isinstance(expr, Im):
            return self._render_complex_component(expr.variable, "imag", None)
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
        if token in {
            "abs",
            "exp",
            "sqrt",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "atan2",
            "floor",
            "sinh",
            "cosh",
            "tanh",
        }:
            self._requires_math = True
            arg0 = expr.arguments[0]
            rendered = self._render_expr_at(arg0, indices) if indices else self._render_expr(arg0)
            if self._is_complex_expr(arg0) and token not in {"atan2", "floor"}:
                return self._render_complex_intrinsic(token, rendered, arg0)
            args = ", ".join(
                self._render_expr_at(arg, indices) if indices else self._render_expr(arg)
                for arg in expr.arguments
            )
            return f"{token}({args})"
        if token in {"real", "aimag"}:
            arg0 = expr.arguments[0]
            rendered = self._render_expr_at(arg0, indices) if indices else self._render_expr(arg0)
            if self._is_complex_expr(arg0):
                suffix = "f" if self._is_complex64(arg0) else ""
                func = "creal" if token == "real" else "cimag"
                return f"{func}{suffix}({rendered})"
            return rendered
        if token == "conjg":
            arg0 = expr.arguments[0]
            rendered = self._render_expr_at(arg0, indices) if indices else self._render_expr(arg0)
            if self._is_complex_expr(arg0):
                suffix = "f" if self._is_complex64(arg0) else ""
                self._requires_math = True
                return f"conj{suffix}({rendered})"
            return rendered
        if token == "cmplx":
            real = expr.arguments[0]
            imag = expr.arguments[1] if len(expr.arguments) > 1 else LiteralNode(0.0)
            real_rendered = (
                self._render_expr_at(real, indices) if indices else self._render_expr(real)
            )
            imag_rendered = (
                self._render_expr_at(imag, indices) if indices else self._render_expr(imag)
            )
            is_float = self._cmplx_is_float(expr)
            self._requires_math = True
            if is_float:
                return f"npy_cpackf((float){real_rendered}, (float){imag_rendered})"
            return f"npy_cpack((double){real_rendered}, (double){imag_rendered})"
        if token in {"iand", "ior", "xor"}:
            left = (
                self._render_expr_at(expr.arguments[0], indices)
                if indices
                else self._render_expr(expr.arguments[0])
            )
            right = (
                self._render_expr_at(expr.arguments[1], indices)
                if indices
                else self._render_expr(expr.arguments[1])
            )
            op = "&" if token == "iand" else ("|" if token == "ior" else "^")
            return f"({left} {op} {right})"
        if token == "ishft":
            value = (
                self._render_expr_at(expr.arguments[0], indices)
                if indices
                else self._render_expr(expr.arguments[0])
            )
            shift = (
                self._render_expr_at(expr.arguments[1], indices)
                if indices
                else self._render_expr(expr.arguments[1])
            )
            return f"(({shift}) >= 0 ? (({value}) << ({shift})) : (({value}) >> (-({shift}))))"
        if token == "ibset":
            value = (
                self._render_expr_at(expr.arguments[0], indices)
                if indices
                else self._render_expr(expr.arguments[0])
            )
            pos = (
                self._render_expr_at(expr.arguments[1], indices)
                if indices
                else self._render_expr(expr.arguments[1])
            )
            return f"(({value}) | (1ULL << ({pos})))"
        if token == "ibclr":
            value = (
                self._render_expr_at(expr.arguments[0], indices)
                if indices
                else self._render_expr(expr.arguments[0])
            )
            pos = (
                self._render_expr_at(expr.arguments[1], indices)
                if indices
                else self._render_expr(expr.arguments[1])
            )
            return f"(({value}) & ~(1ULL << ({pos})))"
        if token == "popcnt":
            value = (
                self._render_expr_at(expr.arguments[0], indices)
                if indices
                else self._render_expr(expr.arguments[0])
            )
            return f"__builtin_popcountll((unsigned long long)({value}))"
        if token == "trailz":
            value = (
                self._render_expr_at(expr.arguments[0], indices)
                if indices
                else self._render_expr(expr.arguments[0])
            )
            width = self._bit_width(expr.arguments[0])
            return f"(({value}) == 0 ? {width} : __builtin_ctzll((unsigned long long)({value})))"
        if token == "allocated":
            arg0 = expr.arguments[0]
            if isinstance(arg0, Variable) and arg0._shape is not SCALAR:
                return f"({arg0.name} != NULL)"
            return "1"
        if token == "rank":
            return str(self._expr_rank_dims(expr.arguments[0])[0])
        if token == "size":
            return self._render_size(expr)
        if token == "shape":
            raise NotImplementedError("shape should be lowered in array context")
        if token in {"max", "min"}:
            return self._render_scalar_max_min(expr)
        if token in {"sum", "maxval", "minval", "all", "dot_product", "matmul", "transpose"}:
            raise NotImplementedError("Array intrinsics should be lowered in array context")
        raise NotImplementedError(f"C backend does not support intrinsic {token}")

    def _render_getitem(self, expr: GetItem, indices: Sequence[str] | None = None) -> str:
        variable = expr.variable
        if not isinstance(variable, (Variable, GetAttr)):
            raise NotImplementedError(
                "C backend only supports getitem on variables or struct fields"
            )
        rank, dims_exprs, fortran_order, base = self._array_expr_info(variable)
        slices = expr.sliced if isinstance(expr.sliced, tuple) else (expr.sliced,)
        slices = list(slices)
        if len(slices) < rank:
            slices.extend([slice(None)] * (rank - len(slices)))
        out_indices: list[str] = []
        index_iter = iter(indices) if indices is not None else None
        for i, slice_ in enumerate(slices):
            if isinstance(slice_, slice):
                if index_iter is None:
                    raise NotImplementedError("C backend requires indices for sliced getitem")
                start, _ = self._slice_bounds(slice_, dims_exprs[i])
                out_indices.append(f"({start} + {next(index_iter)})")
            else:
                idx = self._render_expr(slice_)
                out_indices.append(idx)
        linear = self._linear_index(out_indices, dims_exprs, fortran_order)
        return f"{base}[{linear}]"

    def _render_getattr(self, expr: GetAttr) -> str:
        base = expr.variable
        if isinstance(base, Variable):
            if self._pointer_args.get(base.name, False):
                return f"{base.name}->{expr.attr}"
            return f"{base.name}.{expr.attr}"
        if isinstance(base, GetItem):
            return f"{self._render_getitem(base)}.{expr.attr}"
        if isinstance(base, GetAttr):
            return f"{self._render_getattr(base)}.{expr.attr}"
        raise NotImplementedError("C backend only supports getattr on variables or struct elements")

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
        if isinstance(value, np.complexfloating):
            self._requires_math = True
            real = value.real
            imag = value.imag
            if value.dtype == np.complex64:
                return f"npy_cpackf((float){real}, (float){imag})"
            return f"npy_cpack((double){real}, (double){imag})"
        if isinstance(value, complex):
            self._requires_math = True
            return f"npy_cpack((double){value.real}, (double){value.imag})"
        if isinstance(value, str):
            return f'"{value}"'
        raise NotImplementedError("C backend only supports int/float/bool/complex/string literals")

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

    def _is_complex_expr(self, expr) -> bool:
        ftype = getattr(expr, "_ftype", None)
        return ftype is not None and ftype.type == "complex"

    def _is_complex64(self, expr) -> bool:
        dtype = self._safe_dtype_from_ftype(expr)
        if dtype is None:
            return False
        return dtype.get_numpy() is np.complex64

    def _render_complex_binop(
        self, expr: BinaryOperationNode, indices: Sequence[str] | None
    ) -> str:
        self._requires_math = True
        left = self._render_expr_at(expr.left, indices) if indices else self._render_expr(expr.left)
        right = (
            self._render_expr_at(expr.right, indices) if indices else self._render_expr(expr.right)
        )
        if expr.op == "**":
            func = (
                "cpowf"
                if (self._is_complex64(expr.left) or self._is_complex64(expr.right))
                else "cpow"
            )
            return f"{func}({left}, {right})"
        op = _C_OPS.get(expr.op, expr.op)
        return f"({left} {op} {right})"

    def _render_complex_intrinsic(self, token: str, value: str, arg) -> str:
        self._requires_math = True
        suffix = "f" if self._is_complex64(arg) else ""
        mapping = {
            "abs": f"npy_cabs{suffix}",
            "exp": f"npy_cexp{suffix}",
            "sqrt": f"npy_csqrt{suffix}",
            "sin": f"npy_csin{suffix}",
            "cos": f"npy_ccos{suffix}",
            "tan": f"npy_ctan{suffix}",
            "asin": f"npy_casin{suffix}",
            "acos": f"npy_cacos{suffix}",
            "atan": f"npy_catan{suffix}",
        }
        func = mapping[token]
        return f"{func}({value})"

    def _cmplx_is_float(self, expr: IntrinsicFunction) -> bool:
        if len(expr.arguments) < 3:
            return False
        kind = expr.arguments[2]
        if isinstance(kind, LiteralNode) and isinstance(kind.value, int):
            return kind.value == 4
        return False

    def _render_complex_component(self, base, component: str, indices: Sequence[str] | None) -> str:
        rendered = self._render_expr_at(base, indices) if indices else self._render_expr(base)
        if self._is_complex_expr(base):
            suffix = "f" if self._is_complex64(base) else ""
            func = "creal" if component == "real" else "cimag"
            return f"{func}{suffix}({rendered})"
        return rendered

    def _render_complex_set(self, target, value) -> str:
        base = target.variable
        rhs = self._render_expr(value)
        is_float = self._is_complex64(base)
        macro = "NPY_CSETREALF" if is_float else "NPY_CSETREAL"
        if isinstance(target, Im):
            macro = "NPY_CSETIMAGF" if is_float else "NPY_CSETIMAG"
        if isinstance(base, Variable):
            if self._pointer_args.get(base.name, False):
                ptr = base.name
            else:
                ptr = f"&{base.name}"
        elif isinstance(base, GetItem):
            ptr = f"&{self._render_getitem(base)}"
        else:
            raise NotImplementedError("C backend only supports real/imag assignment on variables")
        return f"{macro}({ptr}, {rhs});"

    def _render_scalar_intrinsic(self, expr: IntrinsicFunction) -> str:
        token = expr.token
        if token == "rank":
            return str(self._expr_rank_dims(expr.arguments[0])[0])
        if token == "size":
            return self._render_size(expr)
        return self._render_intrinsic(expr)

    def _render_size(self, expr: IntrinsicFunction) -> str:
        arg = expr.arguments[0]
        rank, dims_exprs = self._expr_rank_dims(arg)
        if len(expr.arguments) == 1:
            return self._render_product(dims_exprs)
        dim_expr = self._render_expr(expr.arguments[1])
        return self._render_dim_select(dim_expr, dims_exprs, base=1)

    def _render_shape_element(
        self, expr: IntrinsicFunction, index_expr: str
    ) -> tuple[str, list[str]]:
        arg = expr.arguments[0]
        _, dims_exprs = self._expr_rank_dims(arg)
        return (
            self._render_dim_select(index_expr, dims_exprs, base=syntax_settings.array_lower_bound),
            [],
        )

    def _render_dim_select(self, index_expr: str, dims_exprs: Sequence[str], base: int) -> str:
        if len(dims_exprs) == 1:
            return dims_exprs[0]
        parts = []
        for i, dim in enumerate(dims_exprs):
            idx = base + i
            parts.append(f"(({index_expr}) == {idx} ? {dim} : ")
        result = "".join(parts) + "0" + ")" * len(dims_exprs)
        return result

    def _render_reduction(self, expr: IntrinsicFunction, indent: int) -> tuple[str, list[str]]:
        token = expr.token
        if token == "dot_product":
            a = expr.arguments[0]
            b = expr.arguments[1]
            rank, dims_exprs = self._expr_rank_dims(a)
            if rank != 1:
                raise NotImplementedError("dot_product only supports 1-D arrays")
            temp = f"_nm_red{self._temp_counter}"
            self._temp_counter += 1
            ctype = self._ctype_from_expr(a)
            lines = [f"{'    ' * indent}{ctype} {temp} = 0;\n"]
            idx = self._next_loop_var()
            lb = syntax_settings.array_lower_bound
            end = f"{dims_exprs[0]} - 1 + {lb}"
            lines.append(
                f"{'    ' * indent}for (npy_intp {idx} = {lb}; {idx} <= {end}; {idx}++) {{\n"
            )
            lines.append(
                f"{'    ' * (indent + 1)}{temp} += {self._render_expr_at(a, [idx])} * {self._render_expr_at(b, [idx])};\n"
            )
            lines.append(f"{'    ' * indent}}}\n")
            return temp, lines

        arg = expr.arguments[0]
        rank, dims_exprs = self._expr_rank_dims(arg)
        if rank == 0:
            value = self._render_expr(arg)
            if token == "all":
                return value, []
            return value, []
        temp = f"_nm_red{self._temp_counter}"
        self._temp_counter += 1
        ctype = self._ctype_from_expr(arg)
        lines: list[str] = []
        lb = syntax_settings.array_lower_bound
        if token == "sum":
            lines.append(f"{'    ' * indent}{ctype} {temp} = 0;\n")
        elif token == "all":
            lines.append(f"{'    ' * indent}int {temp} = 1;\n")
        else:
            init_indices = [str(lb)] * rank
            init_val = self._render_expr_at(arg, init_indices)
            lines.append(f"{'    ' * indent}{ctype} {temp} = {init_val};\n")

        loop_vars: list[str] = []
        for i in range(rank):
            var = self._next_loop_var()
            loop_vars.append(var)
            end = f"{dims_exprs[i]} - 1 + {lb}"
            lines.append(
                f"{'    ' * indent}for (npy_intp {var} = {lb}; {var} <= {end}; {var}++) {{\n"
            )
            indent += 1

        value = self._render_expr_at(arg, loop_vars)
        if token == "sum":
            lines.append(f"{'    ' * indent}{temp} += {value};\n")
        elif token == "maxval":
            lines.append(f"{'    ' * indent}if ({value} > {temp}) {temp} = {value};\n")
        elif token == "minval":
            lines.append(f"{'    ' * indent}if ({value} < {temp}) {temp} = {value};\n")
        elif token == "all":
            label = f"_nm_all_done_{temp}"
            lines.append(f"{'    ' * indent}if (!({value})) {{ {temp} = 0; goto {label}; }}\n")

        for _ in range(rank):
            indent -= 1
            lines.append(f"{'    ' * indent}}}\n")
        if token == "all":
            label = f"_nm_all_done_{temp}"
            lines.append(f"{label}: ;\n")
        return temp, lines

    def _render_matmul(
        self,
        expr: IntrinsicFunction,
        target_indices: Sequence[str],
        target_dims: Sequence[str],
        indent: int,
    ) -> tuple[str, list[str]]:
        a = expr.arguments[0]
        b = expr.arguments[1]
        a_rank, a_dims = self._expr_rank_dims(a)
        b_rank, b_dims = self._expr_rank_dims(b)
        ctype = self._ctype_from_expr(a)
        temp = f"_nm_mm{self._temp_counter}"
        self._temp_counter += 1
        lines = [f"{'    ' * indent}{ctype} {temp} = 0;\n"]
        k = self._next_loop_var()
        lb = syntax_settings.array_lower_bound
        if a_rank == 1 and b_rank == 1:
            end = f"{a_dims[0]} - 1 + {lb}"
            lines.append(f"{'    ' * indent}for (npy_intp {k} = {lb}; {k} <= {end}; {k}++) {{\n")
            lines.append(
                f"{'    ' * (indent + 1)}{temp} += {self._render_expr_at(a, [k])} * {self._render_expr_at(b, [k])};\n"
            )
            lines.append(f"{'    ' * indent}}}\n")
            return temp, lines
        if a_rank == 2 and b_rank == 2:
            i, j = target_indices
            end = f"{a_dims[1]} - 1 + {lb}"
            lines.append(f"{'    ' * indent}for (npy_intp {k} = {lb}; {k} <= {end}; {k}++) {{\n")
            lines.append(
                f"{'    ' * (indent + 1)}{temp} += {self._render_expr_at(a, [i, k])} * {self._render_expr_at(b, [k, j])};\n"
            )
            lines.append(f"{'    ' * indent}}}\n")
            return temp, lines
        if a_rank == 2 and b_rank == 1:
            i = target_indices[0]
            end = f"{a_dims[1]} - 1 + {lb}"
            lines.append(f"{'    ' * indent}for (npy_intp {k} = {lb}; {k} <= {end}; {k}++) {{\n")
            lines.append(
                f"{'    ' * (indent + 1)}{temp} += {self._render_expr_at(a, [i, k])} * {self._render_expr_at(b, [k])};\n"
            )
            lines.append(f"{'    ' * indent}}}\n")
            return temp, lines
        if a_rank == 1 and b_rank == 2:
            j = target_indices[0]
            end = f"{b_dims[0]} - 1 + {lb}"
            lines.append(f"{'    ' * indent}for (npy_intp {k} = {lb}; {k} <= {end}; {k}++) {{\n")
            lines.append(
                f"{'    ' * (indent + 1)}{temp} += {self._render_expr_at(a, [k])} * {self._render_expr_at(b, [k, j])};\n"
            )
            lines.append(f"{'    ' * indent}}}\n")
            return temp, lines
        raise NotImplementedError("matmul supports 1-D or 2-D arrays only")

    def _render_transpose(
        self, expr: IntrinsicFunction, target_indices: Sequence[str]
    ) -> tuple[str, list[str]]:
        arg = expr.arguments[0]
        if len(target_indices) != 2:
            raise NotImplementedError("transpose only supports 2-D arrays")
        i, j = target_indices
        return self._render_expr_at(arg, [j, i]), []

    def _render_max_min(
        self,
        expr: IntrinsicFunction,
        target_indices: Sequence[str],
        target_dims: Sequence[str],
        indent: int,
    ) -> tuple[str, list[str]]:
        setups: list[str] = []
        rendered_args: list[str] = []
        for arg in expr.arguments:
            rendered, setup = self._render_expr_for_target(arg, target_indices, target_dims, indent)
            setups.extend(setup)
            rendered_args.append(rendered)

        def pick(a, b):
            op = ">" if expr.token == "max" else "<"
            return f"(({a}) {op} ({b}) ? ({a}) : ({b}))"

        current = rendered_args[0]
        for arg in rendered_args[1:]:
            current = pick(current, arg)
        return current, setups

    def _render_scalar_max_min(self, expr: IntrinsicFunction) -> str:
        rendered_args = [self._render_expr(arg) for arg in expr.arguments]

        def pick(a, b):
            op = ">" if expr.token == "max" else "<"
            return f"(({a}) {op} ({b}) ? ({a}) : ({b}))"

        current = rendered_args[0]
        for arg in rendered_args[1:]:
            current = pick(current, arg)
        return current

    def _bit_width(self, expr) -> int:
        dtype = self._safe_dtype_from_ftype(expr)
        if dtype is None:
            return 64
        np_type = dtype.get_numpy()
        return np.dtype(np_type).itemsize * 8

    def _map_ftype_to_ctype(self, variable: Variable) -> str:
        ftype = variable._ftype
        if ftype is None:
            raise NotImplementedError("C backend requires a concrete type")
        if ftype.type == "type" and getattr(ftype.kind, "name", None) == "c_ptr":
            return "void"
        dtype = DataType.from_ftype(ftype)
        return dtype.get_cnumpy()

    def _ctype_from_expr(self, expr) -> str:
        ftype = getattr(expr, "_ftype", None)
        if ftype is None:
            raise NotImplementedError("C backend requires a concrete type")
        if ftype.type == "type" and getattr(ftype.kind, "name", None) == "c_ptr":
            return "void"
        dtype = DataType.from_ftype(ftype)
        return dtype.get_cnumpy()

    def _safe_dtype_from_ftype(self, expr) -> DataType | None:
        ftype = getattr(expr, "_ftype", None)
        if ftype is None:
            return None
        try:
            return DataType.from_ftype(ftype)
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
            if isinstance(expr.value, complex):
                return '"(%g%+gi)"'
        if hasattr(expr, "_ftype"):
            ftype = expr._ftype
            if ftype is not None and ftype.type == "integer":
                return '"%ld"'
            if ftype is not None and ftype.type == "real":
                return '"%g"'
            if ftype is not None and ftype.type == "logical":
                return '"%d"'
            if ftype is not None and ftype.type == "complex":
                return '"(%g%+gi)"'
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
        if isinstance(target, GetAttr):
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
