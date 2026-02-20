from __future__ import annotations

from typing import Any, NoReturn, cast

import numpy as np

from numeta.datatype import DataType
from numeta.exceptions import raise_with_source
from numeta.settings import settings as nm_settings
from numeta.ast.namespace import Namespace
from numeta.ast.variable import Variable

from numeta.c.c_syntax import render_expr_blocks

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
    IRVar,
    IRVarRef,
    IRWhile,
    IROpaqueExpr,
    IROpaqueStmt,
)


_C_BINARY_OPS = {
    "eq": "==",
    "ne": "!=",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "and": "&&",
    "or": "||",
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "pow": "**",
}


class CEmitter:
    def __init__(self) -> None:
        self._array_info: dict[str, dict[str, Any]] = {}
        self._pointer_args: dict[str, bool] = {}
        self._shape_arg_map: dict[str, str] = {}
        self._tmp_counter = 0
        self._pointer_locals: set[str] = set()
        self._requires_math = False
        self._reduction_helpers: dict[tuple[str, str], str] = {}

    @property
    def requires_math(self) -> bool:
        return self._requires_math

    def _resolve_source_node(self, origin: object | None) -> object | None:
        if origin is None:
            return None
        source = getattr(origin, "source", None)
        if source is not None:
            return source
        if getattr(origin, "source_location", None) is not None:
            return origin
        return None

    def _raise_with_source(
        self, exception_class, message: str, origin: object | None = None
    ) -> NoReturn:
        raise_with_source(
            exception_class,
            message,
            source_node=self._resolve_source_node(origin),
        )
        raise AssertionError("unreachable")

    def emit_procedure(self, proc: IRProcedure) -> tuple[str, bool]:
        self._array_info = {}
        self._pointer_args = {}
        self._shape_arg_map = {}
        self._tmp_counter = 0
        self._pointer_locals = set()
        self._requires_math = False
        self._reduction_helpers = {}

        for arg in proc.args:
            if arg.name.startswith("shape_"):
                base = arg.name[len("shape_") :]
                if base:
                    self._shape_arg_map[arg.name] = base

        arg_specs = self._build_signature(proc)
        lines: list[str] = []
        lines.append("#include <Python.h>\n")
        lines.append("#include <numpy/arrayobject.h>\n")
        lines.append("#include <numpy/npy_math.h>\n")
        lines.append("#include <complex.h>\n")
        lines.append("#include <stdlib.h>\n")
        lines.append("#include <omp.h>\n")
        if self._requires_math:
            lines.append("#include <math.h>\n")
        lines.append("\n")

        struct_defs = self._collect_struct_defs(proc)
        if struct_defs:
            lines.extend(struct_defs)
            lines.append("\n")

        reduction_helpers = self._collect_reduction_helpers()
        if reduction_helpers:
            lines.extend(reduction_helpers)
            lines.append("\n")

        global_constants = self._collect_global_constants(proc)
        if global_constants:
            lines.extend(global_constants)
            lines.append("\n")

        prototypes = self._render_prototypes(proc)
        if prototypes:
            lines.extend(prototypes)
            lines.append("\n")

        body_lines = []
        body_lines.extend(self._render_local_declarations(proc, indent=1))
        body_lines.extend(self._render_statements(proc.body, indent=1))

        reduction_helpers = self._collect_reduction_helpers()
        if reduction_helpers:
            lines.extend(reduction_helpers)
            lines.append("\n")

        args = ", ".join(arg_specs)
        lines.append(f"void {proc.name}({args}) {{\n")
        lines.extend(body_lines)
        lines.append("}\n")
        return "".join(lines), self._requires_math

    def _collect_struct_defs(self, proc: IRProcedure) -> list[str]:
        defs: dict[str, str] = {}

        def bfs(dtype: DataType):
            if not dtype.is_struct():
                return
            if dtype.name in defs:
                return
            for _, nested_dtype, _ in dtype.members:
                if nested_dtype.is_struct():
                    bfs(nested_dtype)
            defs[dtype.name] = dtype.c_declaration()

        for var in list(proc.args) + list(proc.locals):
            dtype = self._dtype_from_irvar(var)
            if dtype is not None and dtype.is_struct():
                bfs(dtype)

        return list(defs.values())

    def emit_namespace(self, namespace: Namespace) -> tuple[str, bool]:
        self._array_info = {}
        self._pointer_args = {}
        self._shape_arg_map = {}
        self._tmp_counter = 0
        self._pointer_locals = set()
        self._requires_math = False
        self._reduction_helpers = {}

        lines: list[str] = []
        lines.append("#include <Python.h>\n")
        lines.append("#include <numpy/arrayobject.h>\n")
        lines.append("#include <numpy/npy_math.h>\n")
        lines.append("#include <complex.h>\n")
        lines.append("#include <stdlib.h>\n")
        lines.append("#include <omp.h>\n")
        lines.append("\n")

        # Collect global variables from the module
        for var in namespace.variables.values():
            if var.assign is not None:
                lines.extend(self._render_global_variable(var))

        return "".join(lines), self._requires_math

    def _render_global_variable(self, var: Variable) -> list[str]:
        lines = []
        shape = var._shape
        dtype = var.dtype
        if dtype is None:
            return []
        ctype = dtype.get_cnumpy()
        assign = var.assign

        if shape is None or shape.rank == 0:
            value = self._render_literal(assign)
            lines.append(f"const {ctype} {var.name} = {value};\n")
            return lines

        if shape.dims is None:
            return []

        fortran_order = shape.fortran_order
        values: list[str] = []
        if isinstance(assign, np.ndarray):
            flat = assign.ravel(order="F" if fortran_order else "C")
            for v in flat:
                values.append(self._render_literal(v))
        elif isinstance(assign, (int, float, complex, bool, str, np.generic)):
            values.append(self._render_literal(assign))
        else:
            return []

        if not values:
            return []

        size = len(values)
        lines.append(f"const {ctype} {var.name}[{size}] = {{")
        lines.append(", ".join(values))
        lines.append("};\n")
        return lines

    def _collect_global_constants(self, proc: IRProcedure) -> list[str]:
        constants: dict[str, dict[str, Any]] = {}

        def visit_expr(expr: IRExpr | None):
            if expr is None:
                return
            if isinstance(expr, IRVarRef) and expr.var is not None:
                source = expr.var.source
                if isinstance(source, Variable) and isinstance(source.parent, Namespace):
                    if source.assign is None:
                        return
                    if expr.var.name not in constants:
                        constants[expr.var.name] = {
                            "var": expr.var,
                            "source": source,
                        }
                return
            if isinstance(expr, IRCallExpr):
                visit_expr(expr.callee)
                for arg in expr.args:
                    visit_expr(arg)
                return
            if isinstance(expr, IRGetItem):
                visit_expr(expr.base)
                for idx in expr.indices:
                    if isinstance(idx, IRSlice):
                        visit_expr(idx.start)
                        visit_expr(idx.stop)
                        visit_expr(idx.step)
                    else:
                        visit_expr(cast(IRExpr, idx))
                return
            if isinstance(expr, IRGetAttr):
                visit_expr(expr.base)
                return
            if isinstance(expr, IRBinary):
                visit_expr(expr.left)
                visit_expr(expr.right)
                return
            if isinstance(expr, IRUnary):
                visit_expr(expr.operand)
                return
            if isinstance(expr, IRIntrinsic):
                for arg in expr.args:
                    visit_expr(arg)
                return

        def visit_stmt(stmt: Any):
            if isinstance(stmt, IRAssign):
                visit_expr(stmt.target)
                visit_expr(stmt.value)
            elif isinstance(stmt, IRCall):
                visit_expr(stmt.func)
                for arg in stmt.args:
                    visit_expr(arg)
            elif isinstance(stmt, IRIf):
                visit_expr(stmt.cond)
                for child in stmt.then:
                    visit_stmt(child)
                for child in stmt.else_:
                    visit_stmt(child)
            elif isinstance(stmt, IRFor):
                visit_expr(stmt.start)
                visit_expr(stmt.stop)
                visit_expr(stmt.step)
                for child in stmt.body:
                    visit_stmt(child)
            elif isinstance(stmt, IRWhile):
                visit_expr(stmt.cond)
                for child in stmt.body:
                    visit_stmt(child)
            elif isinstance(stmt, IROpaqueStmt):
                return

        for stmt in proc.body:
            visit_stmt(stmt)

        lines: list[str] = []
        for name, item in constants.items():
            source = item["source"]
            rendered = self._render_global_variable(source)
            if not rendered:
                continue

            # Since _render_global_variable is generic, we might need to adjust for static/const
            # if we want strictly static const inside the function scope,
            # but usually global constants are extern/global.
            # However, the original code used `static const`.
            # Let's adjust the rendered lines to be `static const` if they are not already.

            # Actually, reusing _render_global_variable which emits `const ...`
            # and prefixing `static` here is tricky with list of strings.
            # But wait, the previous code emitted `static const`.
            # My new _render_global_variable emits `const`.
            # I should probably pass a modifier to _render_global_variable.

            lines.extend([l.replace("const ", "static const ", 1) for l in rendered])

            # We also need to populate self._array_info for arrays so they can be used
            # This logic was inside the loop before.
            shape = source._shape
            if shape is not None and shape.rank > 0 and shape.dims is not None:
                # We need to reconstruct what _render_global_variable did to get array info
                # or better, have _render_global_variable populate it if asked?
                # Or just manually populate it here since we have the source.

                dtype = source.dtype
                if dtype is None:
                    continue
                ctype = dtype.get_cnumpy()
                dims_exprs = [self._render_dim(dim) for dim in shape.dims]
                self._array_info[name] = {
                    "name": name,
                    "ctype": ctype,
                    "rank": len(dims_exprs),
                    "fortran_order": shape.fortran_order,
                    "dims_exprs": dims_exprs,
                    "dims_name": None,
                }

        return lines

    def _render_prototypes(self, proc: IRProcedure) -> list[str]:
        names = self._collect_callees(proc.body)
        names.discard(proc.name)
        return [f"void {name}(...);\n" for name in sorted(names)]

    def _collect_callees(self, statements: list[Any]) -> set[str]:
        names: set[str] = set()

        def visit_expr(expr: IRExpr | None):
            if expr is None:
                return
            if isinstance(expr, IRCallExpr):
                visit_expr(expr.callee)
                for arg in expr.args:
                    visit_expr(arg)
                return
            if isinstance(expr, IRGetItem):
                visit_expr(expr.base)
                for idx in expr.indices:
                    if isinstance(idx, IRSlice):
                        visit_expr(idx.start)
                        visit_expr(idx.stop)
                        visit_expr(idx.step)
                    else:
                        visit_expr(cast(IRExpr, idx))
                return
            if isinstance(expr, IRGetAttr):
                visit_expr(expr.base)
                return
            if isinstance(expr, IRBinary):
                visit_expr(expr.left)
                visit_expr(expr.right)
                return
            if isinstance(expr, IRUnary):
                visit_expr(expr.operand)
                return
            if isinstance(expr, IRIntrinsic):
                for arg in expr.args:
                    visit_expr(arg)
                return

        for stmt in statements:
            if isinstance(stmt, IRCall):
                if isinstance(stmt.func, IRVarRef):
                    names.add(self._call_name(stmt.func))
                visit_expr(stmt.func)
                for arg in stmt.args:
                    visit_expr(arg)
            elif isinstance(stmt, IRIf):
                visit_expr(stmt.cond)
                names |= self._collect_callees(stmt.then)
                names |= self._collect_callees(stmt.else_)
            elif isinstance(stmt, IRFor):
                visit_expr(stmt.start)
                visit_expr(stmt.stop)
                visit_expr(stmt.step)
                names |= self._collect_callees(stmt.body)
            elif isinstance(stmt, IRWhile):
                visit_expr(stmt.cond)
                names |= self._collect_callees(stmt.body)
            elif isinstance(stmt, IRAssign):
                visit_expr(stmt.target)
                visit_expr(stmt.value)

        return names

    def _dtype_from_irvar(self, var: IRVar) -> DataType | None:
        source: Any = var.source
        if source is None:
            return None
        dtype = getattr(source, "dtype", None)
        if dtype is None:
            return None
        return dtype

    def _build_signature(self, proc: IRProcedure) -> list[str]:
        arg_specs: list[str] = []
        for var in proc.args:
            if var.name in self._shape_arg_map:
                continue
            shape = var.vtype.shape if var.vtype else None
            if shape is None:
                ctype = self._map_irvar_to_ctype(var)
                dtype = self._dtype_from_irvar(var)
                pass_by_value = dtype is not None and var.intent == "in" and dtype.can_be_value()
                is_pointer = not pass_by_value
                self._pointer_args[var.name] = is_pointer
                const_prefix = "const " if (var.intent == "in" and is_pointer) else ""
                ptr = "*" if is_pointer else ""
                arg_specs.append(f"{const_prefix}{ctype} {ptr}{var.name}")
                continue

            rank = shape.rank or 1
            fortran_order = shape.order == "F"
            dims_name = None
            has_shape_descriptor = nm_settings.add_shape_descriptors and any(
                not isinstance(dim, int) for dim in (shape.dims or [])
            )
            dims_exprs: list[str] = []
            if has_shape_descriptor or f"shape_{var.name}" in self._shape_arg_map:
                dims_name = f"{var.name}_dims"
                arg_specs.append(f"npy_intp* {dims_name}")
                dims_exprs = [f"{dims_name}[{i}]" for i in range(rank)]
            else:
                dims_exprs = [self._render_dim(dim) for dim in (shape.dims or [])]
            if not dims_exprs:
                dims_exprs = ["1"] * rank

            ctype = self._map_irvar_to_ctype(var)
            arg_specs.append(f"{ctype}* {var.name}")
            self._array_info[var.name] = {
                "name": var.name,
                "ctype": ctype,
                "rank": rank,
                "fortran_order": fortran_order,
                "dims_exprs": dims_exprs,
                "dims_name": dims_name,
            }

        return arg_specs

    def _render_local_declarations(self, proc: IRProcedure, indent: int) -> list[str]:
        lines: list[str] = []
        for var in proc.locals:
            shape = var.vtype.shape if var.vtype else None
            ctype = self._map_irvar_to_ctype(var)
            const_prefix = "const " if var.parameter else ""
            if shape is None:
                if var.pointer:
                    lines.append(f"{'    ' * indent}{const_prefix}{ctype} *{var.name} = NULL;\n")
                else:
                    init = ""
                    if var.assign is not None:
                        init = f" = {self._render_literal(var.assign)}"
                    lines.append(f"{'    ' * indent}{const_prefix}{ctype} {var.name}{init};\n")
                continue

            rank = shape.rank or 1
            fortran_order = shape.order == "F"
            dims_name = f"{var.name}_dims"
            dims_exprs = [self._render_dim(dim) for dim in (shape.dims or [])]
            self._array_info[var.name] = {
                "name": var.name,
                "ctype": ctype,
                "rank": rank,
                "fortran_order": fortran_order,
                "dims_exprs": dims_exprs,
                "dims_name": dims_name,
            }
            if var.allocatable or var.pointer or shape.dims is None:
                lines.append(f"{'    ' * indent}{ctype} *{var.name} = NULL;\n")
                continue

            total = self._render_product(dims_exprs)
            lines.append(f"{'    ' * indent}{ctype} {var.name}[{total}];\n")

        if lines:
            lines.append("\n")
        return lines

    def _render_statements(self, statements: list[Any], indent: int) -> list[str]:
        lines: list[str] = []
        for stmt in statements:
            lines.extend(self._render_statement(stmt, indent=indent))
        return lines

    def _render_statement(self, stmt: Any, indent: int) -> list[str]:
        if isinstance(stmt, IRAssign):
            if isinstance(stmt.target, IROpaqueExpr) and stmt.target.payload is not None:
                payload = stmt.target.payload
                payload_type = getattr(payload, "__class__", None)
                payload_name = getattr(payload_type, "__name__", "")
                if payload_name in {"Re", "Im"}:
                    base = getattr(payload, "variable", None)
                    if (
                        getattr(base, "__class__", None) is not None
                        and base.__class__.__name__ == "GetItem"
                    ):
                        base_expr = self._render_getitem_ast(base)
                    else:
                        blocks = render_expr_blocks(base, shape_arg_map=self._shape_arg_map)
                        base_expr = "".join(str(b) for b in blocks)
                    target = (
                        f"__real__({base_expr})"
                        if payload_name == "Re"
                        else f"__imag__({base_expr})"
                    )
                    value = self._render_expr(stmt.value)
                    return [f"{'    ' * indent}{target} = {value};\n"]
            if isinstance(stmt.target, IRGetItem) and any(
                isinstance(idx, IRSlice) for idx in stmt.target.indices
            ):
                return self._render_slice_assignment(
                    stmt.target, cast(IRExpr, stmt.value or IRLiteral(value=0)), indent
                )
            if isinstance(stmt.target, IRGetAttr):
                shape = stmt.target.vtype.shape if stmt.target.vtype else None
                if shape is not None and shape.rank:
                    return self._render_attr_array_assignment(stmt.target, stmt.value, indent)
            if isinstance(stmt.target, IRVarRef) and stmt.target.var is not None:
                name = stmt.target.var.name
                info = self._array_info.get(name)
                if info is not None and info.get("rank", 0) > 0:
                    return self._render_array_assignment(name, stmt.value, indent)
                if isinstance(stmt.value, IRIntrinsic) and stmt.value.name == "dot_product":
                    return self._render_dot_product_assignment(stmt.target, stmt.value, indent)
            target = self._render_expr(stmt.target)
            value = self._render_expr(stmt.value)
            return [f"{'    ' * indent}{target} = {value};\n"]
        if isinstance(stmt, IRCall):
            if isinstance(stmt.func, IRVarRef) and stmt.func.var is not None:
                if stmt.func.var.name == "numpy_allocate":
                    return self._render_numpy_allocate(stmt, indent)
                if stmt.func.var.name == "c_f_pointer":
                    return self._render_c_f_pointer(stmt, indent)
                if not getattr(stmt.func.var.source, "bind_c", True):
                    return self._render_fortran_call(stmt, indent)
            if any(
                (isinstance(arg, IRGetItem) and any(isinstance(i, IRSlice) for i in arg.indices))
                or (isinstance(arg, IRIntrinsic) and arg.name == "matmul")
                or (
                    isinstance(arg, IRGetAttr)
                    and arg.vtype is not None
                    and arg.vtype.shape is not None
                    and arg.vtype.shape.rank
                )
                for arg in stmt.args
            ):
                return self._render_call_with_slices(stmt, indent)
            call = self._render_call(stmt)
            return [f"{'    ' * indent}{call};\n"]
        if isinstance(stmt, IRIf):
            cond = self._render_expr(stmt.cond)
            lines = [f"{'    ' * indent}if ({cond}) {{\n"]
            lines.extend(self._render_statements(stmt.then, indent + 1))
            if stmt.else_:
                idx = 0
                while idx < len(stmt.else_):
                    branch = stmt.else_[idx]
                    if (
                        isinstance(branch, IRIf)
                        and getattr(branch.source, "__class__", None) is not None
                    ):
                        if branch.source.__class__.__name__ == "ElseIf":
                            cond_branch = self._render_expr(branch.cond)
                            lines.append(f"{'    ' * indent}}} else if ({cond_branch}) {{\n")
                            lines.extend(self._render_statements(branch.then, indent + 1))
                            idx += 1
                            continue
                    lines.append(f"{'    ' * indent}}} else {{\n")
                    lines.extend(self._render_statements(stmt.else_[idx:], indent + 1))
                    break
            lines.append(f"{'    ' * indent}}}\n")
            return lines
        if isinstance(stmt, IRFor):
            iterator = stmt.var.name if stmt.var is not None else "i"
            start = self._render_expr(stmt.start)
            end = self._render_expr(stmt.stop)
            step = self._render_expr(stmt.step) if stmt.step is not None else "1"
            condition = "<="
            if isinstance(stmt.step, IRLiteral) and isinstance(stmt.step.value, (int, float)):
                if stmt.step.value < 0:
                    condition = ">="
            init = f"{iterator} = {start}"
            cond = f"{iterator} {condition} {end}"
            incr = f"{iterator} += {step}" if step != "1" else f"{iterator}++"
            lines = [f"{'    ' * indent}for ({init}; {cond}; {incr}) {{\n"]
            lines.extend(self._render_statements(stmt.body, indent + 1))
            lines.append(f"{'    ' * indent}}}\n")
            return lines
        if isinstance(stmt, IRWhile):
            cond = self._render_expr(stmt.cond)
            lines = [f"{'    ' * indent}while ({cond}) {{\n"]
            lines.extend(self._render_statements(stmt.body, indent + 1))
            lines.append(f"{'    ' * indent}}}\n")
            return lines
        if isinstance(stmt, IRAllocate):
            if not isinstance(stmt.var, IRVarRef):
                self._raise_with_source(
                    NotImplementedError,
                    "C backend allocate supports variable targets",
                    origin=stmt,
                )
            var = stmt.var.var
            if var is None:
                self._raise_with_source(
                    NotImplementedError,
                    "C backend allocate requires named variables",
                    origin=stmt,
                )
            name = var.name
            info = self._array_info.get(name)
            if info is None:
                return []
            dims = [self._render_expr(dim) for dim in stmt.dims]
            size = self._render_product(dims)
            lines = []
            if info.get("dims_name"):
                for i, dim in enumerate(dims):
                    lines.append(f"{'    ' * indent}{info['dims_name']}[{i}] = {dim};\n")
            lines.append(
                f"{'    ' * indent}{name} = ({info['ctype']}*)malloc(sizeof({info['ctype']}) * {size});\n"
            )
            return lines
        if isinstance(stmt, IRDeallocate):
            target = self._render_expr(stmt.var)
            return [f"{'    ' * indent}free({target});\n"]
        if isinstance(stmt, IRReturn):
            if stmt.value is None:
                return [f"{'    ' * indent}return;\n"]
            return [f"{'    ' * indent}return {self._render_expr(stmt.value)};\n"]
        if isinstance(stmt, IROpaqueStmt):
            payload = stmt.payload
            if payload is not None and payload.__class__.__name__ == "PointerAssignment":
                pointer = getattr(payload, "pointer", None)
                shape = getattr(payload, "pointer_shape", None)
                target = getattr(payload, "target", None)
                if pointer is not None and shape is not None:
                    pointer_shape = shape
                    pointer_shape_info = getattr(pointer, "_shape", None)
                    if pointer_shape_info is not None and not pointer_shape_info.fortran_order:
                        pointer_shape = tuple(reversed(pointer_shape))
                    name = pointer.name
                    dims_exprs = [self._render_slice_length(dim) for dim in pointer_shape]
                    ctype = self._map_irvar_to_ctype(IRVar(name=name, vtype=None, source=pointer))
                    self._array_info[name] = {
                        "name": name,
                        "ctype": ctype,
                        "rank": len(dims_exprs),
                        "fortran_order": bool(getattr(pointer_shape_info, "fortran_order", False)),
                        "dims_exprs": dims_exprs,
                        "dims_name": None,
                    }
                if pointer is not None and target is not None:
                    target_name = getattr(target, "name", None)
                    if target_name is None:
                        target_name = str(target)
                    return [f"{'    ' * indent}{pointer.name} = {target_name};\n"]
                return []
            return [f"{'    ' * indent}/* unsupported statement */\n"]
        return [f"{'    ' * indent}/* unsupported statement */\n"]

    def _render_numpy_allocate(self, stmt: IRCall, indent: int) -> list[str]:
        if len(stmt.args) < 2:
            self._raise_with_source(
                NotImplementedError,
                "numpy_allocate requires pointer and size",
                origin=stmt,
            )
        ptr_arg = stmt.args[0]
        size_arg = stmt.args[1]
        self._tmp_counter += 1
        size_name = f"_nm_size_{self._tmp_counter}"
        size_expr = self._render_expr(size_arg)
        pre = f"{'    ' * indent}size_t {size_name} = {size_expr};\n"
        call = f"{'    ' * indent}numpy_allocate({self._render_call_arg(ptr_arg)}, &{size_name});\n"
        return [pre, call]

    def _call_name(self, func: IRExpr) -> str:
        name = self._render_expr(func)
        if isinstance(func, IRVarRef) and func.var is not None:
            source = func.var.source
            bind_c = getattr(source, "bind_c", True)
            if not bind_c and not name.endswith("_"):
                name = f"{name}_"
        return name

    def _render_call(self, stmt: IRCall) -> str:
        name = self._call_name(stmt.func)
        args = ", ".join(self._render_call_arg(arg) for arg in stmt.args)
        return f"{name}({args})"

    def _render_fortran_call(self, stmt: IRCall, indent: int) -> list[str]:
        pre_lines: list[str] = []
        call_args: list[str] = []

        for arg in stmt.args:
            if isinstance(arg, IRVarRef) and arg.var is not None:
                if arg.var.vtype is not None and arg.var.vtype.shape is not None:
                    call_args.append(self._render_call_arg(arg))
                else:
                    call_args.append(f"&{arg.var.name}")
                continue
            if isinstance(arg, (IRGetItem, IRGetAttr)):
                call_args.append(f"&({self._render_expr(arg)})")
                continue

            self._tmp_counter += 1
            tmp_name = f"_nm_arg_{self._tmp_counter}"
            ctype = "double"
            expr_value = None
            if isinstance(arg, IRLiteral):
                if isinstance(arg.value, int):
                    ctype = "long long"
                elif isinstance(arg.value, float):
                    ctype = "double"
                elif isinstance(arg.value, str):
                    if len(arg.value) == 1:
                        ctype = "char"
                        expr_value = f"'{arg.value}'"
            elif arg.vtype is not None:
                ctype = self._map_irvar_to_ctype(IRVar(name=tmp_name, vtype=arg.vtype))
            if expr_value is None:
                expr_value = self._render_expr(arg)
            pre_lines.append(f"{'    ' * indent}{ctype} {tmp_name} = {expr_value};\n")
            call_args.append(f"&{tmp_name}")

        call = f"{'    ' * indent}{self._call_name(stmt.func)}({', '.join(call_args)});\n"
        return pre_lines + [call]

    def _render_call_with_slices(self, stmt: IRCall, indent: int) -> list[str]:
        name = self._render_expr(stmt.func)
        pre_lines: list[str] = []
        post_lines: list[str] = []
        call_args: list[str] = []

        def needs_implicit_shape_descriptor(arg: IRExpr) -> bool:
            if arg.vtype is None or arg.vtype.shape is None:
                return False
            dims = arg.vtype.shape.dims or []
            for dim in dims:
                if isinstance(dim, int):
                    continue
                if isinstance(dim, IRLiteral) and isinstance(dim.value, int):
                    continue
                return True
            return False

        def has_explicit_shape_descriptor(arg_idx: int) -> bool:
            if arg_idx == 0:
                return False
            prev = stmt.args[arg_idx - 1]
            if isinstance(prev, IRIntrinsic) and prev.name == "array_constructor":
                return True
            if isinstance(prev, IRVarRef) and prev.var is not None:
                return prev.var.name.startswith("shape_")
            return False

        for idx, arg in enumerate(stmt.args):
            if isinstance(arg, IRGetItem) and any(isinstance(i, IRSlice) for i in arg.indices):
                temp_name, pre, post = self._materialize_slice(arg, indent)
                pre_lines.extend(pre)
                post_lines.extend(post)
                dims = self._shape_dims_for_expr(arg)
                if (
                    dims
                    and needs_implicit_shape_descriptor(arg)
                    and not has_explicit_shape_descriptor(idx)
                ):
                    call_args.append(f"(npy_intp[]){{{', '.join(dims)}}}")
                call_args.append(temp_name)
                continue
            if isinstance(arg, IRIntrinsic) and arg.name == "matmul":
                temp_name, pre, post = self._materialize_matmul(arg, indent)
                pre_lines.extend(pre)
                post_lines.extend(post)
                dims = self._shape_dims_for_expr(arg)
                if (
                    dims
                    and needs_implicit_shape_descriptor(arg)
                    and not has_explicit_shape_descriptor(idx)
                ):
                    call_args.append(f"(npy_intp[]){{{', '.join(dims)}}}")
                call_args.append(temp_name)
                continue
            call_args.append(self._render_call_arg(arg))

        call_line = f"{'    ' * indent}{name}({', '.join(call_args)});\n"
        return pre_lines + [call_line] + post_lines

    def _materialize_slice(self, arg: IRGetItem, indent: int) -> tuple[str, list[str], list[str]]:
        base_name = self._render_expr(arg.base)
        info = self._array_info.get(base_name)
        if info is None:
            self._raise_with_source(
                NotImplementedError,
                "C backend requires array metadata for slice calls",
                origin=arg,
            )

        ctype = info["ctype"]
        dims_exprs = info["dims_exprs"]
        fortran_order = info["fortran_order"]
        rank = info["rank"]

        slice_dims: list[str] = []
        src_indices: list[str] = []
        loop_indices: list[str] = []

        for dim_idx, idx in enumerate(arg.indices):
            loop_var = f"_nm_i{dim_idx}"
            loop_indices.append(loop_var)
            if isinstance(idx, IRSlice):
                start = self._render_expr(idx.start) if idx.start is not None else "0"
                step = self._render_expr(idx.step) if idx.step is not None else "1"
                if idx.stop is None:
                    stop = dims_exprs[dim_idx]
                else:
                    stop = self._render_expr(idx.stop)
                if idx.step is None or step == "1":
                    length = f"({stop}) - ({start})"
                    src_idx = f"({start}) + {loop_var}"
                else:
                    length = f"(({stop}) - ({start}) + ({step}) - 1) / ({step})"
                    src_idx = f"({start}) + ({step}) * {loop_var}"
                slice_dims.append(length)
                src_indices.append(src_idx)
            else:
                slice_dims.append("1")
                src_indices.append(self._render_expr(cast(IRExpr, idx)))

        self._tmp_counter += 1
        temp_name = f"_nm_tmp_{self._tmp_counter}"
        size = self._render_product(slice_dims)
        pre_lines = [
            f"{'    ' * indent}{ctype} *{temp_name} = ({ctype}*)malloc(sizeof({ctype}) * {size});\n"
        ]

        pre_lines.extend(
            self._render_slice_copy(
                base_name,
                temp_name,
                src_indices,
                loop_indices,
                slice_dims,
                dims_exprs,
                fortran_order,
                indent,
            )
        )

        post_lines = []
        post_lines.extend(
            self._render_slice_copy(
                base_name,
                temp_name,
                src_indices,
                loop_indices,
                slice_dims,
                dims_exprs,
                fortran_order,
                indent,
                reverse=True,
            )
        )
        post_lines.append(f"{'    ' * indent}free({temp_name});\n")

        return temp_name, pre_lines, post_lines

    def _render_slice_copy(
        self,
        base_name: str,
        temp_name: str,
        src_indices: list[str],
        loop_indices: list[str],
        slice_dims: list[str],
        dims_exprs: list[str],
        fortran_order: bool,
        indent: int,
        reverse: bool = False,
    ) -> list[str]:
        lines: list[str] = []
        loop_indent = indent
        for loop_var, dim in zip(loop_indices, slice_dims):
            lines.append(
                f"{'    ' * loop_indent}for (npy_intp {loop_var} = 0; {loop_var} < {dim}; {loop_var}++) {{\n"
            )
            loop_indent += 1

        src_linear = self._linear_index(src_indices, dims_exprs, fortran_order)
        dst_linear = self._linear_index(loop_indices, slice_dims, False)
        if reverse:
            lines.append(
                f"{'    ' * loop_indent}({base_name})[{src_linear}] = ({temp_name})[{dst_linear}];\n"
            )
        else:
            lines.append(
                f"{'    ' * loop_indent}({temp_name})[{dst_linear}] = ({base_name})[{src_linear}];\n"
            )

        for _ in loop_indices:
            loop_indent -= 1
            lines.append(f"{'    ' * loop_indent}}}\n")

        return lines

    def _materialize_matmul(
        self, expr: IRIntrinsic, indent: int
    ) -> tuple[str, list[str], list[str]]:
        if len(expr.args) != 2:
            self._raise_with_source(
                NotImplementedError,
                "C backend matmul expects two arguments",
                origin=expr,
            )
        left, right = expr.args
        if not isinstance(left, IRExpr) or not isinstance(right, IRExpr):
            self._raise_with_source(
                NotImplementedError,
                "C backend matmul requires array expressions",
                origin=expr,
            )

        shape = expr.vtype.shape if expr.vtype else None
        if shape is None or shape.rank != 2:
            self._raise_with_source(
                NotImplementedError,
                "C backend matmul requires 2D result",
                origin=expr,
            )
        if shape.dims is None:
            self._raise_with_source(
                NotImplementedError,
                "C backend matmul requires known dims",
                origin=expr,
            )

        dims = [self._render_dim(dim) for dim in shape.dims]
        if len(dims) != 2:
            self._raise_with_source(
                NotImplementedError,
                "C backend matmul requires 2D dims",
                origin=expr,
            )

        self._tmp_counter += 1
        temp_name = f"_nm_tmp_{self._tmp_counter}"
        size = self._render_product(dims)
        ctype = self._map_irvar_to_ctype(IRVar(name=temp_name, vtype=expr.vtype))

        pre_lines = [
            f"{'    ' * indent}{ctype} *{temp_name} = ({ctype}*)malloc(sizeof({ctype}) * {size});\n"
        ]

        i_var = f"_nm_i{self._tmp_counter}"
        j_var = f"_nm_j{self._tmp_counter}"
        k_var = f"_nm_k{self._tmp_counter}"

        pre_lines.append(
            f"{'    ' * indent}for (npy_intp {i_var} = 0; {i_var} < {dims[0]}; {i_var}++) {{\n"
        )
        pre_lines.append(
            f"{'    ' * (indent + 1)}for (npy_intp {j_var} = 0; {j_var} < {dims[1]}; {j_var}++) {{\n"
        )
        pre_lines.append(f"{'    ' * (indent + 2)}{ctype} acc = 0;\n")

        left_info = None
        right_info = None
        if isinstance(left, IRVarRef) and left.var is not None:
            left_info = self._array_info.get(left.var.name)
        if isinstance(right, IRVarRef) and right.var is not None:
            right_info = self._array_info.get(right.var.name)
        if left_info is None or right_info is None:
            self._raise_with_source(
                NotImplementedError,
                "C backend matmul requires array variables",
                origin=expr,
            )

        if not (left_info["fortran_order"] or right_info["fortran_order"]):
            left_info, right_info = right_info, left_info

        pre_lines.append(
            f"{'    ' * (indent + 2)}for (npy_intp {k_var} = 0; {k_var} < {left_info['dims_exprs'][1]}; {k_var}++) {{\n"
        )

        left_linear = self._linear_index(
            [i_var, k_var], left_info["dims_exprs"], left_info["fortran_order"]
        )
        right_linear = self._linear_index(
            [k_var, j_var], right_info["dims_exprs"], right_info["fortran_order"]
        )
        pre_lines.append(
            f"{'    ' * (indent + 3)}acc += ({left_info['name']})[{left_linear}] * ({right_info['name']})[{right_linear}];\n"
        )
        pre_lines.append(f"{'    ' * (indent + 2)}}}\n")

        dst_linear = self._linear_index([i_var, j_var], dims, False)
        pre_lines.append(f"{'    ' * (indent + 2)}({temp_name})[{dst_linear}] = acc;\n")
        pre_lines.append(f"{'    ' * (indent + 1)}}}\n")
        pre_lines.append(f"{'    ' * indent}}}\n")

        post_lines = [f"{'    ' * indent}free({temp_name});\n"]
        return temp_name, pre_lines, post_lines

    def _render_call_arg(self, arg: IRExpr) -> str:
        if isinstance(arg, IRGetItem):
            return f"&({self._render_getitem(arg)})"
        if isinstance(arg, IRGetAttr):
            return f"&({self._render_expr(arg)})"
        if isinstance(arg, IRVarRef) and arg.var is not None:
            name = arg.var.name
            if name in self._shape_arg_map:
                return f"{self._shape_arg_map[name]}_dims"
            if self._pointer_args.get(name, False):
                return name
            if arg.var.intent in {"out", "inout"} and (
                arg.var.vtype is None or arg.var.vtype.shape is None
            ):
                return f"&{name}"
            if name.startswith("fc_") and (arg.var.vtype is None or arg.var.vtype.shape is None):
                return f"&{name}"
        return self._render_expr(arg)

    def _render_c_f_pointer(self, stmt: IRCall, indent: int) -> list[str]:
        if len(stmt.args) < 2:
            self._raise_with_source(
                NotImplementedError,
                "c_f_pointer requires source and target",
                origin=stmt,
            )
        source = stmt.args[0]
        target = stmt.args[1]
        if not isinstance(target, IRVarRef) or target.var is None:
            self._raise_with_source(
                NotImplementedError,
                "c_f_pointer target must be variable",
                origin=stmt,
            )

        target_name = target.var.name
        if isinstance(source, IRCallExpr) and self._render_expr(source.callee) == "c_loc":
            arg0 = source.args[0] if source.args else None
            if isinstance(arg0, IRGetItem):
                source_expr = f"&({self._render_getitem(arg0)})"
            elif isinstance(arg0, IRVarRef) and arg0.var is not None:
                source_expr = arg0.var.name
            else:
                source_expr = self._render_expr(arg0)
        else:
            source_expr = self._render_expr(source)
        ctype = self._map_irvar_to_ctype(target.var)
        lines = [f"{'    ' * indent}{target_name} = ({ctype}*)({source_expr});\n"]

        self._pointer_locals.add(target_name)

        shape_arg = stmt.args[2] if len(stmt.args) > 2 else None
        if shape_arg is not None:
            dims_exprs: list[str] = []
            if isinstance(shape_arg, IRIntrinsic) and shape_arg.name == "array_constructor":
                dims_exprs = [self._render_expr(arg) for arg in shape_arg.args]
            else:
                dims_exprs = [self._render_expr(shape_arg)]
            if dims_exprs:
                self._array_info[target_name] = {
                    "name": target_name,
                    "ctype": ctype,
                    "rank": len(dims_exprs),
                    "fortran_order": False,
                    "dims_exprs": dims_exprs,
                    "dims_name": None,
                }
                return lines

        src_arg = None
        if isinstance(source, IRCallExpr) and source.args:
            src_arg = source.args[0]
        else:
            src_arg = source

        if isinstance(src_arg, IRVarRef) and src_arg.var is not None:
            src_info = self._array_info.get(src_arg.var.name)
            if src_info is not None:
                src_dtype = self._dtype_from_irvar(src_arg.var)
                target_dtype = self._dtype_from_irvar(target.var)
                if src_dtype is not None and target_dtype is not None:
                    try:
                        src_bytes = src_dtype.get_nbytes()
                        target_bytes = target_dtype.get_nbytes()
                    except (AttributeError, TypeError):
                        src_bytes = None
                        target_bytes = None
                    if (
                        src_bytes is not None
                        and target_bytes is not None
                        and src_bytes != target_bytes
                    ):
                        total = self._render_product(src_info["dims_exprs"])
                        scaled = f"(({total})*{src_bytes})/({target_bytes})"
                        self._array_info[target_name] = {
                            "name": target_name,
                            "ctype": ctype,
                            "rank": 1,
                            "fortran_order": False,
                            "dims_exprs": [scaled],
                            "dims_name": None,
                        }
                        return lines
                self._array_info[target_name] = {
                    "name": target_name,
                    "ctype": ctype,
                    "rank": src_info["rank"],
                    "fortran_order": src_info["fortran_order"],
                    "dims_exprs": src_info["dims_exprs"],
                    "dims_name": None,
                }

        return lines

    def _render_array_assignment(self, name: str, value: IRExpr | None, indent: int) -> list[str]:
        info = self._array_info.get(name)
        if info is None:
            return []
        rank = info.get("rank", 0)
        if rank == 0:
            return [f"{'    ' * indent}{name} = {self._render_expr(value)};\n"]

        if isinstance(value, IRIntrinsic) and value.name == "shape" and value.args:
            dims = self._shape_dims_for_expr(value.args[0])
            if dims:
                lines = []
                for idx, dim in enumerate(dims):
                    lines.append(f"{'    ' * indent}({name})[{idx}] = {dim};\n")
                return lines

        if isinstance(value, IRIntrinsic) and value.name == "transpose":
            return self._render_transpose_assignment(name, info, value, indent)
        if isinstance(value, IRIntrinsic) and value.name == "matmul":
            return self._render_matmul_assignment(name, info, value, indent)

        indices = [f"i_{i}" for i in range(rank)]
        lines: list[str] = []
        loop_indent = indent
        for idx, dim in zip(indices, info["dims_exprs"]):
            lines.append(
                f"{'    ' * loop_indent}for (npy_intp {idx} = 0; {idx} < {dim}; {idx}++) {{\n"
            )
            loop_indent += 1

        linear = self._linear_index(indices, info["dims_exprs"], info["fortran_order"])
        if isinstance(value, IRIntrinsic) and value.name == "array_constructor":
            ctor_vals = ", ".join(self._render_expr(arg) for arg in value.args)
            rhs = f"(({info['ctype']}[]){{{ctor_vals}}})[{linear}]"
        else:
            rhs = self._render_expr_with_slice(value, indices)

        lines.append(f"{'    ' * loop_indent}({name})[{linear}] = {rhs};\n")

        for _ in range(rank):
            loop_indent -= 1
            lines.append(f"{'    ' * loop_indent}}}\n")

        return lines

    def _render_dot_product_assignment(
        self, target: IRVarRef, expr: IRIntrinsic, indent: int
    ) -> list[str]:
        if len(expr.args) != 2:
            self._raise_with_source(
                NotImplementedError,
                "dot_product expects two args",
                origin=expr,
            )
        left, right = expr.args
        if not (isinstance(left, IRVarRef) and isinstance(right, IRVarRef)):
            self._raise_with_source(
                NotImplementedError,
                "dot_product requires array variables",
                origin=expr,
            )
        left_info = self._array_info.get(left.var.name if left.var else "")
        right_info = self._array_info.get(right.var.name if right.var else "")
        if left_info is None or right_info is None:
            self._raise_with_source(
                NotImplementedError,
                "dot_product requires array metadata",
                origin=expr,
            )

        loop_var = "_nm_i_dot"
        dim = left_info["dims_exprs"][0]
        ctype = left_info["ctype"]
        target_expr = self._render_expr(target)

        lines = [f"{'    ' * indent}{ctype} acc = 0;\n"]
        lines.append(
            f"{'    ' * indent}for (npy_intp {loop_var} = 0; {loop_var} < {dim}; {loop_var}++) {{\n"
        )
        left_linear = self._linear_index(
            [loop_var], left_info["dims_exprs"], left_info["fortran_order"]
        )
        right_linear = self._linear_index(
            [loop_var], right_info["dims_exprs"], right_info["fortran_order"]
        )
        lines.append(
            f"{'    ' * (indent + 1)}acc += ({left.var.name})[{left_linear}] * ({right.var.name})[{right_linear}];\n"
        )
        lines.append(f"{'    ' * indent}}}\n")
        lines.append(f"{'    ' * indent}{target_expr} = acc;\n")
        return lines

    def _render_transpose_assignment(
        self, name: str, info: dict[str, Any], expr: IRIntrinsic, indent: int
    ) -> list[str]:
        if len(expr.args) != 1:
            self._raise_with_source(
                NotImplementedError,
                "transpose expects one arg",
                origin=expr,
            )
        src = expr.args[0]
        if not isinstance(src, IRVarRef) or src.var is None:
            self._raise_with_source(
                NotImplementedError,
                "transpose requires array variable",
                origin=expr,
            )
        src_info = self._array_info.get(src.var.name)
        if src_info is None:
            self._raise_with_source(
                NotImplementedError,
                "transpose requires array metadata",
                origin=expr,
            )
        if info.get("rank", 0) != 2:
            self._raise_with_source(
                NotImplementedError,
                "transpose requires rank-2 arrays",
                origin=expr,
            )

        i_var = "_nm_i_t"
        j_var = "_nm_j_t"
        lines = [
            f"{'    ' * indent}for (npy_intp {i_var} = 0; {i_var} < {info['dims_exprs'][0]}; {i_var}++) {{\n",
            f"{'    ' * (indent + 1)}for (npy_intp {j_var} = 0; {j_var} < {info['dims_exprs'][1]}; {j_var}++) {{\n",
        ]
        dst_linear = self._linear_index([i_var, j_var], info["dims_exprs"], info["fortran_order"])
        src_linear = self._linear_index(
            [j_var, i_var], src_info["dims_exprs"], src_info["fortran_order"]
        )
        lines.append(
            f"{'    ' * (indent + 2)}({name})[{dst_linear}] = ({src.var.name})[{src_linear}];\n"
        )
        lines.append(f"{'    ' * (indent + 1)}}}\n")
        lines.append(f"{'    ' * indent}}}\n")
        return lines

    def _render_matmul_assignment(
        self, name: str, info: dict[str, Any], expr: IRIntrinsic, indent: int
    ) -> list[str]:
        if len(expr.args) != 2:
            self._raise_with_source(
                NotImplementedError,
                "matmul expects two args",
                origin=expr,
            )
        left, right = expr.args
        if not (isinstance(left, IRVarRef) and isinstance(right, IRVarRef)):
            self._raise_with_source(
                NotImplementedError,
                "matmul requires array variables",
                origin=expr,
            )

        left_info = self._array_info.get(left.var.name if left.var else "")
        right_info = self._array_info.get(right.var.name if right.var else "")
        if left_info is None or right_info is None:
            self._raise_with_source(
                NotImplementedError,
                "matmul requires array metadata",
                origin=expr,
            )

        if not (left_info["fortran_order"] or right_info["fortran_order"]):
            left_info, right_info = right_info, left_info

        i_var = "_nm_i_m"
        j_var = "_nm_j_m"
        k_var = "_nm_k_m"
        ctype = info["ctype"]
        lines = [
            f"{'    ' * indent}for (npy_intp {i_var} = 0; {i_var} < {info['dims_exprs'][0]}; {i_var}++) {{\n",
            f"{'    ' * (indent + 1)}for (npy_intp {j_var} = 0; {j_var} < {info['dims_exprs'][1]}; {j_var}++) {{\n",
            f"{'    ' * (indent + 2)}{ctype} acc = 0;\n",
            f"{'    ' * (indent + 2)}for (npy_intp {k_var} = 0; {k_var} < {left_info['dims_exprs'][1]}; {k_var}++) {{\n",
        ]
        left_linear = self._linear_index(
            [i_var, k_var], left_info["dims_exprs"], left_info["fortran_order"]
        )
        right_linear = self._linear_index(
            [k_var, j_var], right_info["dims_exprs"], right_info["fortran_order"]
        )
        lines.append(
            f"{'    ' * (indent + 3)}acc += ({left_info['name']})[{left_linear}] * ({right_info['name']})[{right_linear}];\n"
        )
        lines.append(f"{'    ' * (indent + 2)}}}\n")
        dst_linear = self._linear_index([i_var, j_var], info["dims_exprs"], info["fortran_order"])
        lines.append(f"{'    ' * (indent + 2)}({name})[{dst_linear}] = acc;\n")
        lines.append(f"{'    ' * (indent + 1)}}}\n")
        lines.append(f"{'    ' * indent}}}\n")
        return lines

    def _render_slice_assignment(
        self, target: IRGetItem, value: IRExpr | None, indent: int
    ) -> list[str]:
        base_name = self._render_expr(target.base)
        info = self._array_info.get(base_name)
        if info is None:
            self._raise_with_source(
                NotImplementedError,
                "C backend requires array metadata for slice assignment",
                origin=target,
            )

        indices: list[str] = []
        loops: list[tuple[str, str, str, str]] = []
        for dim_idx, idx in enumerate(target.indices):
            if isinstance(idx, IRSlice):
                var = f"i_{dim_idx}"
                start = self._render_expr(idx.start) if idx.start is not None else "0"
                stop = (
                    self._render_expr(idx.stop)
                    if idx.stop is not None
                    else info["dims_exprs"][dim_idx]
                )
                step = self._render_expr(idx.step) if idx.step is not None else "1"
                loops.append((var, start, stop, step))
                indices.append(var)
            else:
                indices.append(self._render_expr(cast(IRExpr, idx)))

        loop_indices = [var for var, _, _, _ in loops]
        linear = self._linear_index(indices, info["dims_exprs"], info["fortran_order"])
        value_expr = self._render_expr_with_slice(value, loop_indices)

        lines: list[str] = []
        loop_indent = indent
        for var, start, stop, step in loops:
            condition = "<"
            if step.startswith("-"):
                condition = ">"
            init = f"npy_intp {var} = {start}"
            cond = f"{var} {condition} {stop}"
            incr = f"{var} += {step}" if step != "1" else f"{var}++"
            lines.append(f"{'    ' * loop_indent}for ({init}; {cond}; {incr}) {{\n")
            loop_indent += 1

        lines.append(f"{'    ' * loop_indent}({base_name})[{linear}] = {value_expr};\n")

        for _ in loops:
            loop_indent -= 1
            lines.append(f"{'    ' * loop_indent}}}\n")

        return lines

    def _render_expr_with_slice(self, expr: IRExpr | None, loop_vars: list[str]) -> str:
        if expr is None:
            return ""
        if isinstance(expr, IRVarRef) and expr.var is not None:
            name = expr.var.name
            info = self._array_info.get(name)
            if info is not None and loop_vars:
                linear = self._linear_index(loop_vars, info["dims_exprs"], info["fortran_order"])
                return f"({name})[{linear}]"
        if isinstance(expr, IRGetItem):
            if not any(isinstance(i, IRSlice) for i in expr.indices):
                return self._render_getitem(expr)
            base_name = self._render_expr(expr.base)
            info = self._array_info.get(base_name)
            if info is None:
                return self._render_getitem(expr)
            indices: list[str] = []
            loop_iter = iter(loop_vars)
            for idx in expr.indices:
                if isinstance(idx, IRSlice):
                    start = self._render_expr(idx.start) if idx.start is not None else "0"
                    loop_var = next(loop_iter, "0")
                    indices.append(f"({start}) + {loop_var}")
                else:
                    indices.append(self._render_expr(cast(IRExpr, idx)))
            linear = self._linear_index(indices, info["dims_exprs"], info["fortran_order"])
            return f"({base_name})[{linear}]"
        if isinstance(expr, IRBinary):
            op = _C_BINARY_OPS.get(expr.op, expr.op)
            if op == "**":
                self._requires_math = True
                return f"pow({self._render_expr_with_slice(expr.left, loop_vars)}, {self._render_expr_with_slice(expr.right, loop_vars)})"
            return f"({self._render_expr_with_slice(expr.left, loop_vars)} {op} {self._render_expr_with_slice(expr.right, loop_vars)})"
        if isinstance(expr, IRUnary):
            if expr.op == "neg":
                return f"(-{self._render_expr_with_slice(expr.operand, loop_vars)})"
            if expr.op == "not":
                return f"(!{self._render_expr_with_slice(expr.operand, loop_vars)})"
        if isinstance(expr, IRCallExpr):
            name = self._render_expr(expr.callee)
            if name == "c_loc" and expr.args:
                arg0 = expr.args[0]
                if isinstance(arg0, IRVarRef) and arg0.var is not None:
                    return arg0.var.name
                return self._render_expr(arg0)
            args = ", ".join(self._render_expr_with_slice(arg, loop_vars) for arg in expr.args)
            return f"{name}({args})"
        return self._render_expr(expr)

    def _render_expr(self, expr: IRExpr | None) -> str:
        if expr is None:
            return ""
        if isinstance(expr, IRLiteral):
            return self._render_literal(expr.value)
        if isinstance(expr, IRVarRef):
            if expr.var is None:
                return ""
            name = expr.var.name
            if name in self._shape_arg_map:
                return f"{self._shape_arg_map[name]}_dims"
            if self._pointer_args.get(name, False):
                return f"(*{name})"
            if name in self._pointer_locals:
                return f"(*{name})"
            return name
        if isinstance(expr, IRBinary):
            op = _C_BINARY_OPS.get(expr.op, expr.op)
            if op == "**":
                self._requires_math = True
                return f"pow({self._render_expr(expr.left)}, {self._render_expr(expr.right)})"
            return f"({self._render_expr(expr.left)} {op} {self._render_expr(expr.right)})"
        if isinstance(expr, IRUnary):
            if expr.op == "neg":
                return f"(-{self._render_expr(expr.operand)})"
            if expr.op == "not":
                return f"(!{self._render_expr(expr.operand)})"
        if isinstance(expr, IRCallExpr):
            name = self._render_expr(expr.callee)
            if name == "c_loc" and expr.args:
                arg0 = expr.args[0]
                if isinstance(arg0, IRVarRef) and arg0.var is not None:
                    return arg0.var.name
                return self._render_expr(arg0)
            args = ", ".join(self._render_expr(arg) for arg in expr.args)
            return f"{name}({args})"
        if isinstance(expr, IRGetAttr):
            return self._render_attr_access(expr)
        if isinstance(expr, IRGetItem):
            return self._render_getitem(expr)
        if isinstance(expr, IRIntrinsic):
            return self._render_intrinsic(expr)
        if isinstance(expr, IROpaqueExpr):
            if expr.payload is not None:
                payload = expr.payload
                payload_type = getattr(payload, "__class__", None)
                payload_name = getattr(payload_type, "__name__", "")
                if payload_name == "GetItem":
                    return self._render_getitem_ast(payload)
                if payload_name == "GetAttr":
                    attr = getattr(payload, "attr", None)
                    base = getattr(payload, "variable", None)
                    if attr in {"real", "imag"}:
                        base_expr = ""
                        if (
                            getattr(base, "__class__", None) is not None
                            and base.__class__.__name__ == "GetItem"
                        ):
                            base_expr = self._render_getitem_ast(base)
                        else:
                            blocks = render_expr_blocks(base, shape_arg_map=self._shape_arg_map)
                            base_expr = "".join(str(b) for b in blocks)
                        func = "creal" if attr == "real" else "cimag"
                        return f"{func}({base_expr})"
                if payload_name in {"Re", "Im"}:
                    base = getattr(payload, "variable", None)
                    base_expr = ""
                    if (
                        getattr(base, "__class__", None) is not None
                        and base.__class__.__name__ == "GetItem"
                    ):
                        base_expr = self._render_getitem_ast(base)
                    else:
                        blocks = render_expr_blocks(base, shape_arg_map=self._shape_arg_map)
                        base_expr = "".join(str(b) for b in blocks)
                    func = "creal" if payload_name == "Re" else "cimag"
                    return f"{func}({base_expr})"
                blocks = render_expr_blocks(payload, shape_arg_map=self._shape_arg_map)
                return "".join(str(b) for b in blocks)
            return ""
        return ""

    def _render_getitem(self, expr: IRGetItem) -> str:
        if not isinstance(expr.base, IRExpr):
            self._raise_with_source(
                NotImplementedError,
                "C backend only supports variable array base",
                origin=expr,
            )
        base_name = self._render_expr(expr.base)
        if isinstance(expr.base, IRVarRef) and expr.base.var is not None:
            base_var_name = expr.base.var.name
            if base_var_name in self._shape_arg_map:
                if not expr.indices:
                    return base_name
                idx0 = expr.indices[0]
                if isinstance(idx0, IRSlice):
                    self._raise_with_source(
                        NotImplementedError,
                        "C backend does not support slicing shape dims",
                        origin=expr,
                    )
                return f"{base_name}[{self._render_expr(cast(IRExpr, idx0))}]"
        info = self._array_info.get(base_name)
        if info is None:
            if isinstance(expr.base, IRVarRef) and expr.base.var is not None:
                shape = expr.base.var.vtype.shape if expr.base.var.vtype is not None else None
                if shape is not None and shape.dims is not None:
                    dims_exprs = [self._render_dim(dim) for dim in shape.dims]
                    indices: list[str] = []
                    for idx in expr.indices:
                        if isinstance(idx, IRSlice):
                            start_expr = idx.start
                            if start_expr is None:
                                indices.append("0")
                            else:
                                indices.append(self._render_expr(start_expr))
                            continue
                        indices.append(self._render_expr(cast(IRExpr, idx)))
                    linear = self._linear_index(indices, dims_exprs, shape.order == "F")
                    return f"({base_name})[{linear}]"
            if isinstance(expr.base, IRGetAttr):
                shape = expr.base.vtype.shape if expr.base.vtype is not None else None
                if shape is not None and shape.dims is not None:
                    dims_exprs = [self._render_dim(dim) for dim in shape.dims]
                    indices = []
                    for idx in expr.indices:
                        if isinstance(idx, IRSlice):
                            start_expr = idx.start
                            if start_expr is None:
                                indices.append("0")
                            else:
                                indices.append(self._render_expr(start_expr))
                            continue
                        indices.append(self._render_expr(cast(IRExpr, idx)))
                    linear = self._linear_index(indices, dims_exprs, shape.order == "F")
                    return f"({base_name})[{linear}]"
            self._raise_with_source(
                NotImplementedError,
                "C backend requires array metadata for getitem",
                origin=expr,
            )
        indices: list[str] = []
        for idx in expr.indices:
            if isinstance(idx, IRSlice):
                start_expr = idx.start
                if start_expr is None:
                    indices.append("0")
                else:
                    indices.append(self._render_expr(start_expr))
                continue
            indices.append(self._render_expr(cast(IRExpr, idx)))
        linear = self._linear_index(indices, info["dims_exprs"], info["fortran_order"])
        return f"({base_name})[{linear}]"

    def _render_getitem_ast(self, expr: Any) -> str:
        base = getattr(expr, "variable", None)
        base_name = getattr(base, "name", None)
        if base_name is None:
            return ""
        info = self._array_info.get(base_name)
        if info is None:
            return base_name

        indices: list[str] = []
        sliced = getattr(expr, "sliced", None)
        if isinstance(sliced, tuple):
            elements = list(sliced)
        else:
            elements = [sliced]

        for idx in elements:
            if isinstance(idx, slice):
                start = idx.start
                indices.append("0" if start is None else self._render_dim(start))
            else:
                indices.append(self._render_dim(idx))

        linear = self._linear_index(indices, info["dims_exprs"], info["fortran_order"])
        return f"({base_name})[{linear}]"

    def _shape_dims_for_expr(self, expr: IRExpr | None) -> list[str]:
        if isinstance(expr, IRExpr) and expr.vtype is not None and expr.vtype.shape is not None:
            shape = expr.vtype.shape
            if shape.dims is not None:
                if any(dim is None for dim in shape.dims):
                    if isinstance(expr, IRVarRef) and expr.var is not None:
                        info = self._array_info.get(expr.var.name)
                        if info is not None:
                            return list(info["dims_exprs"])
                rendered = [self._render_dim(dim) for dim in shape.dims]
                if any(not dim for dim in rendered):
                    if isinstance(expr, IRVarRef) and expr.var is not None:
                        info = self._array_info.get(expr.var.name)
                        if info is not None:
                            return list(info["dims_exprs"])
                return rendered
        if isinstance(expr, IRVarRef) and expr.var is not None:
            info = self._array_info.get(expr.var.name)
            if info is not None:
                return list(info["dims_exprs"])
        if isinstance(expr, IRGetItem):
            base_name = self._render_expr(expr.base)
            info = self._array_info.get(base_name)
            if info is None:
                return []
            dims: list[str] = []
            for dim_idx, idx in enumerate(expr.indices):
                if isinstance(idx, IRSlice):
                    start = self._render_expr(idx.start) if idx.start is not None else "0"
                    step = self._render_expr(idx.step) if idx.step is not None else "1"
                    if idx.stop is None:
                        stop = info["dims_exprs"][dim_idx]
                    else:
                        stop = self._render_expr(idx.stop)
                    if idx.step is None or step == "1":
                        length = f"({stop}) - ({start})"
                    else:
                        length = f"(({stop}) - ({start}) + ({step}) - 1) / ({step})"
                    dims.append(length)
            return dims
        return []

    def _render_attr_access(self, expr: IRGetAttr) -> str:
        base = expr.base
        if isinstance(base, IRVarRef) and base.var is not None:
            name = base.var.name
            if self._pointer_args.get(name, False) or name in self._pointer_locals:
                return f"{name}->{expr.name}"
            return f"{name}.{expr.name}"
        return f"{self._render_expr(base)}.{expr.name}"

    def _render_attr_array_assignment(
        self, target: IRGetAttr, value: IRExpr | None, indent: int
    ) -> list[str]:
        shape = target.vtype.shape if target.vtype else None
        if shape is None or shape.rank is None:
            return []
        dims = [self._render_dim(dim) for dim in shape.dims or []]
        if not dims:
            return [
                f"{'    ' * indent}{self._render_attr_access(target)} = {self._render_expr(value)};\n"
            ]

        base_expr = self._render_attr_access(target)
        if len(dims) > 1:
            base_ptr = f"(&({base_expr})[0][0])"
        else:
            base_ptr = f"({base_expr})"
        indices = [f"i_{i}" for i in range(len(dims))]
        lines: list[str] = []
        loop_indent = indent
        for idx, dim in zip(indices, dims):
            lines.append(
                f"{'    ' * loop_indent}for (npy_intp {idx} = 0; {idx} < {dim}; {idx}++) {{\n"
            )
            loop_indent += 1

        linear = self._linear_index(indices, dims, False)
        rhs = self._render_expr(value)
        lines.append(f"{'    ' * loop_indent}{base_ptr}[{linear}] = {rhs};\n")

        for _ in indices:
            loop_indent -= 1
            lines.append(f"{'    ' * loop_indent}}}\n")

        return lines

    def _render_slice_length(self, slice_) -> str:
        if not isinstance(slice_, slice):
            return self._render_dim(slice_)
        start = slice_.start
        stop = slice_.stop
        if stop is None:
            return "1"
        start_expr = "0" if start is None else self._render_dim(start)
        stop_expr = self._render_dim(stop)
        return f"({stop_expr}) - ({start_expr})"

    def _register_reduction_helper(self, name: str, ctype: str) -> str:
        key = (name, ctype)
        if key in self._reduction_helpers:
            return self._reduction_helpers[key]
        helper_name = f"nm_{name}_{ctype.replace('*', 'ptr').replace(' ', '_')}"
        self._reduction_helpers[key] = helper_name
        return helper_name

    def _collect_reduction_helpers(self) -> list[str]:
        lines: list[str] = []
        for (name, ctype), helper_name in self._reduction_helpers.items():
            if name == "all":
                lines.append(
                    f"static npy_bool {helper_name}(const {ctype}* data, npy_intp size) {{\n"
                )
                lines.append("    for (npy_intp i = 0; i < size; i++) {\n")
                lines.append("        if (!data[i]) {\n")
                lines.append("            return 0;\n")
                lines.append("        }\n")
                lines.append("    }\n")
                lines.append("    return 1;\n")
                lines.append("}\n")
                continue

            lines.append(f"static {ctype} {helper_name}(const {ctype}* data, npy_intp size) {{\n")
            lines.append("    if (size <= 0) {\n")
            lines.append(f"        return ({ctype})0;\n")
            lines.append("    }\n")
            lines.append(f"    {ctype} acc = data[0];\n")
            lines.append("    for (npy_intp i = 1; i < size; i++) {\n")
            if name == "sum":
                lines.append("        acc += data[i];\n")
            elif name == "maxval":
                lines.append("        if (data[i] > acc) {\n")
                lines.append("            acc = data[i];\n")
                lines.append("        }\n")
            elif name == "minval":
                lines.append("        if (data[i] < acc) {\n")
                lines.append("            acc = data[i];\n")
                lines.append("        }\n")
            lines.append("    }\n")
            lines.append("    return acc;\n")
            lines.append("}\n")
        return lines

    def _is_complex_expr(self, expr: IRExpr) -> bool:
        if isinstance(expr, IRVarRef) and expr.var is not None:
            dtype = self._dtype_from_irvar(expr.var)
            if dtype is not None:
                return dtype.get_numpy() in (np.complex64, np.complex128)
        if expr.vtype is not None and expr.vtype.dtype is not None:
            return expr.vtype.dtype.name == "complex"
        return False

    def _is_integer_expr(self, expr: IRExpr) -> bool:
        if isinstance(expr, IRLiteral) and isinstance(expr.value, int):
            return True
        if expr.vtype and expr.vtype.dtype.name == "integer":
            return True
        if (
            isinstance(expr, IRVarRef)
            and expr.var
            and self._dtype_from_irvar(expr.var)
            and self._dtype_from_irvar(expr.var).name == "integer"
        ):
            return True
        return False

    def _render_intrinsic(self, expr: IRIntrinsic) -> str:
        name = expr.name
        args = [self._render_expr(arg) for arg in expr.args]
        if name == "array_constructor":
            return f"(npy_intp[]){{{', '.join(args)}}}"
        if name == "shape":
            if not expr.args:
                return "(npy_intp[]){0}"
            dims = self._shape_dims_for_expr(expr.args[0])
            if dims:
                return f"(npy_intp[]){{{', '.join(dims)}}}"
            return "(npy_intp[]){0}"
        if name == "size":
            if not expr.args:
                return "0"
            base = expr.args[0]
            dim_index = None
            if len(expr.args) > 1 and isinstance(expr.args[1], IRLiteral):
                try:
                    dim_index = int(expr.args[1].value)
                except (ValueError, TypeError):
                    dim_index = None
            if isinstance(base, IRVarRef) and base.var is not None:
                info = self._array_info.get(base.var.name)
                if info is not None:
                    if dim_index is not None and 1 <= dim_index <= len(info["dims_exprs"]):
                        return info["dims_exprs"][dim_index - 1]
                    return self._render_product(info["dims_exprs"])
            return "0"
        if name == "rank":
            if not expr.args:
                return "0"
            base = expr.args[0]
            if isinstance(base, IRVarRef) and base.var is not None:
                info = self._array_info.get(base.var.name)
                if info is not None:
                    return str(info.get("rank", 0))
            return "0"
        if name in {"sum", "maxval", "minval", "all"}:
            if not expr.args:
                return "0"
            base = expr.args[0]
            if isinstance(base, IRVarRef) and base.var is not None:
                info = self._array_info.get(base.var.name)
                if info is not None:
                    total = self._render_product(info["dims_exprs"])
                    helper_name = self._register_reduction_helper(name, info["ctype"])
                    return f"{helper_name}({base.var.name}, {total})"
            return "0"
        if name == "abs":
            self._requires_math = True
            if not expr.args:
                return "abs()"
            arg0 = expr.args[0]
            if self._is_complex_expr(arg0):
                return f"cabs({args[0]})"

            is_int = False
            if isinstance(arg0, IRVarRef) and arg0.var is not None:
                dtype = self._dtype_from_irvar(arg0.var)
                if dtype and dtype.name == "integer":
                    is_int = True
            if not is_int and arg0.vtype and arg0.vtype.dtype.name == "integer":
                is_int = True
            if not is_int and isinstance(arg0, IRLiteral) and isinstance(arg0.value, int):
                is_int = True

            if is_int:
                return f"llabs({args[0]})"
            return f"fabs({args[0]})"

        if name == "log10":
            self._requires_math = True
            is_complex = any(self._is_complex_expr(arg) for arg in expr.args)

            arg0_str = args[0]
            if self._is_integer_expr(expr.args[0]):
                from numeta.datatype import float64

                dtype = float64
                ctype = dtype.get_cnumpy()
                arg0_str = f"({ctype})({arg0_str})"

            if is_complex:
                # Workaround for missing clog10 in C99: log10(z) = log(z)/log(10)
                return f"(clog({arg0_str})/log(10.0))"
            return f"log10({arg0_str})"

        if name in {
            "exp",
            "sqrt",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "tanh",
            "asinh",
            "acosh",
            "atanh",
            "log",
        }:
            self._requires_math = True
            is_complex = any(self._is_complex_expr(arg) for arg in expr.args)
            if is_complex:
                return f"c{name}({', '.join(args)})"

            # Cast integer arguments to DEFAULT_FLOAT (e.g. float64) to ensure consistency
            # with numeta settings, rather than relying on implicit C int->double promotion.
            new_args = []
            for i, arg_expr in enumerate(expr.args):
                arg_str = args[i]
                if self._is_integer_expr(arg_expr):
                    from numeta.datatype import float64

                    dtype = float64
                    ctype = dtype.get_cnumpy()
                    arg_str = f"({ctype})({arg_str})"
                new_args.append(arg_str)
            return f"{name}({', '.join(new_args)})"

        if name in {"atan2", "floor", "ceil", "hypot", "copysign"}:
            self._requires_math = True
            new_args = []
            for i, arg_expr in enumerate(expr.args):
                arg_str = args[i]
                if self._is_integer_expr(arg_expr):
                    from numeta.datatype import float64

                    dtype = float64
                    ctype = dtype.get_cnumpy()
                    arg_str = f"({ctype})({arg_str})"
                new_args.append(arg_str)
            return f"{name}({', '.join(new_args)})"
        if name == "real":
            if expr.args and self._is_complex_expr(expr.args[0]):
                return f"creal({args[0]})"
            return args[0]
        if name == "aimag":
            if expr.args and self._is_complex_expr(expr.args[0]):
                return f"cimag({args[0]})"
            return args[0]
        if name == "conjg":
            self._requires_math = True
            return f"conj({args[0]})"
        if name in {"iand", "ior", "xor"}:
            op = "&" if name == "iand" else ("|" if name == "ior" else "^")
            return f"({args[0]} {op} {args[1]})"
        if name == "ishft":
            return f"(({args[1]}) >= 0 ? (({args[0]}) << ({args[1]})) : (({args[0]}) >> (-({args[1]}))))"
        if name == "ibset":
            return f"(({args[0]}) | (1ULL << ({args[1]})))"
        if name == "ibclr":
            return f"(({args[0]}) & ~(1ULL << ({args[1]})))"
        if name == "popcnt":
            return f"__builtin_popcountll((unsigned long long)({args[0]}))"
        if name == "trailz":
            return f"(({args[0]}) == 0 ? 64 : __builtin_ctzll((unsigned long long)({args[0]})))"
        return f"{name}({', '.join(args)})"

    def _map_irvar_to_ctype(self, var: IRVar) -> str:
        source: Any = var.source
        if source is not None:
            dtype = getattr(source, "dtype", None)
            if dtype is not None:
                return dtype.get_cnumpy()
        # Fallback
        return "double"

    def _render_dim(self, dim: Any) -> str:
        if isinstance(dim, int):
            return str(dim)
        if isinstance(dim, IRExpr):
            return self._render_expr(dim)
        if hasattr(dim, "variable") and hasattr(dim, "sliced"):
            base = getattr(dim.variable, "name", None)
            if base in self._shape_arg_map:
                index = dim.sliced
                if isinstance(index, int):
                    return f"{self._shape_arg_map[base]}_dims[{index}]"
        blocks = render_expr_blocks(dim, shape_arg_map=self._shape_arg_map)
        if blocks:
            return "".join(str(b) for b in blocks)
        return str(dim)

    def _render_product(self, terms: list[str]) -> str:
        if not terms:
            return "1"
        result = terms[0]
        for term in terms[1:]:
            result = f"({result})*({term})"
        return result

    def _linear_index(self, indices: list[str], dims: list[str], fortran_order: bool) -> str:
        if not indices:
            return "0"
        if len(indices) != len(dims):
            return indices[0]
        if fortran_order:
            result = indices[0]
            stride = dims[0]
            for idx, dim in zip(indices[1:], dims[1:]):
                result = f"({result}) + ({stride})*({idx})"
                stride = f"({stride})*({dim})"
            return result
        result = indices[-1]
        stride = dims[-1]
        for idx, dim in zip(reversed(indices[:-1]), reversed(dims[:-1])):
            result = f"({result}) + ({stride})*({idx})"
            stride = f"({stride})*({dim})"
        return result

    def _render_literal(self, value: Any) -> str:
        if isinstance(value, str):
            return f'"{value}"'
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, complex):
            return f"({value.real} + {value.imag}*I)"
        if isinstance(value, np.generic):
            return str(value)
        return str(value)
