from __future__ import annotations

from typing import Any

from numeta.array_shape import SCALAR, UNKNOWN
from numeta.ast import Variable
from numeta.ast.settings import settings as syntax_settings
from numeta.ast.expressions import (
    BinaryOperationNode,
    FunctionCall,
    GetAttr,
    GetItem,
    IntrinsicFunction,
    LiteralNode,
)
from numeta.ast.expressions.various import ArrayConstructor
from numeta.ast.statements import (
    Allocate,
    Assignment,
    Call,
    Case,
    Deallocate,
    Do,
    DoWhile,
    Else,
    ElseIf,
    If,
    Return,
    SelectCase,
)
from numeta.ast.statements.variable_declaration import VariableDeclaration
from numeta.ast.subroutine import Subroutine
from numeta.ast.function import Function

from .nodes import (
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
    IRNode,
    IRProcedure,
    IRReturn,
    IRShape,
    IRSlice,
    IRType,
    IRUnary,
    IRValueType,
    IRVar,
    IRVarRef,
    IRWhile,
    IROpaqueExpr,
    IROpaqueStmt,
)


_BINARY_OPS: dict[str, str] = {
    ".eq.": "eq",
    ".ne.": "ne",
    ".lt.": "lt",
    ".le.": "le",
    ".gt.": "gt",
    ".ge.": "ge",
    ".and.": "and",
    ".or.": "or",
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "**": "pow",
}


def lower_subroutine(subroutine: Subroutine) -> IRProcedure:
    var_cache: dict[int, IRVar] = {}

    def lower_var(var: Variable, *, is_arg: bool) -> IRVar:
        key = id(var)
        if key in var_cache:
            return var_cache[key]
        vtype = _lower_value_type(var._ftype, var._shape)
        storage = "value"
        if getattr(var, "allocatable", False):
            storage = "allocatable"
        elif getattr(var, "pointer", False):
            storage = "pointer"
        ir_var = IRVar(
            name=var.name,
            vtype=vtype,
            intent=var.intent,
            storage=storage,
            is_const=var.intent == "in",
            is_arg=is_arg,
            allocatable=getattr(var, "allocatable", False),
            pointer=getattr(var, "pointer", False),
            target=getattr(var, "target", False),
            parameter=getattr(var, "parameter", False),
            bind_c=getattr(var, "bind_c", False),
            assign=getattr(var, "assign", None),
            source=var,
        )
        var_cache[key] = ir_var
        return ir_var

    def lower_expr(expr) -> IRExpr:
        if isinstance(expr, IRExpr):
            return expr
        if isinstance(expr, (Subroutine, Function)):
            return IRVarRef(var=IRVar(name=expr.name, source=expr), source=expr)
        if isinstance(expr, LiteralNode):
            return IRLiteral(
                value=expr.value,
                vtype=_lower_value_type(expr._ftype, expr._shape),
                source=expr,
            )
        if isinstance(expr, Variable):
            return IRVarRef(
                var=lower_var(expr, is_arg=expr.name in subroutine.arguments),
                vtype=_lower_value_type(expr._ftype, expr._shape),
                source=expr,
            )
        if isinstance(expr, BinaryOperationNode):
            op = _map_binary_op(expr.op)
            return IRBinary(
                op=op,
                left=lower_expr(expr.left),
                right=lower_expr(expr.right),
                vtype=_lower_value_type(expr._ftype, expr._shape),
                source=expr,
            )
        if isinstance(expr, FunctionCall):
            return IRCallExpr(
                callee=lower_expr(expr.function),
                args=[lower_expr(arg) for arg in expr.arguments],
                vtype=_lower_value_type(expr._ftype, expr._shape),
                source=expr,
            )
        if isinstance(expr, GetItem):
            indices = _lower_indices(expr.sliced)
            shape = _safe_shape(expr)
            return IRGetItem(
                base=lower_expr(expr.variable),
                indices=indices,
                vtype=_lower_value_type(expr._ftype, shape),
                source=expr,
            )
        if isinstance(expr, GetAttr):
            return IRGetAttr(
                base=lower_expr(expr.variable),
                name=expr.attr,
                vtype=_lower_value_type(expr._ftype, expr._shape),
                source=expr,
            )
        if isinstance(expr, ArrayConstructor):
            return IRIntrinsic(
                name="array_constructor",
                args=[lower_expr(arg) for arg in expr.elements],
                vtype=_lower_value_type(expr._ftype, expr._shape),
                source=expr,
            )
        if isinstance(expr, IntrinsicFunction):
            token = getattr(expr, "token", "")
            args = [lower_expr(arg) for arg in expr.arguments]
            if token == "-" and len(args) == 1:
                return IRUnary(
                    op="neg",
                    operand=args[0],
                    vtype=_lower_value_type(expr._ftype, expr._shape),
                    source=expr,
                )
            if token == ".not." and len(args) == 1:
                return IRUnary(
                    op="not",
                    operand=args[0],
                    vtype=_lower_value_type(expr._ftype, expr._shape),
                    source=expr,
                )
            return IRIntrinsic(
                name=token,
                args=args,
                vtype=_lower_value_type(expr._ftype, expr._shape),
                source=expr,
            )
        return IROpaqueExpr(payload=expr, source=expr)

    def lower_stmt(stmt) -> IRNode:
        if isinstance(stmt, Assignment):
            return IRAssign(
                target=lower_expr(stmt.target),
                value=lower_expr(stmt.value),
                source=stmt,
            )
        if isinstance(stmt, Call):
            return IRCall(
                func=lower_expr(stmt.function),
                args=[lower_expr(arg) for arg in stmt.arguments],
                source=stmt,
            )
        if isinstance(stmt, If):
            then_body = [lower_stmt(s) for s in stmt.scope.get_statements()]
            else_body: list[IRNode] = []
            for branch in stmt.orelse:
                if isinstance(branch, ElseIf):
                    nested = IRIf(
                        cond=lower_expr(branch.condition),
                        then=[lower_stmt(s) for s in branch.scope.get_statements()],
                        else_=[],
                        source=branch,
                    )
                    else_body.append(nested)
                elif isinstance(branch, Else):
                    else_body.extend([lower_stmt(s) for s in branch.scope.get_statements()])
                else:
                    else_body.append(lower_stmt(branch))
            return IRIf(
                cond=lower_expr(stmt.condition),
                then=then_body,
                else_=else_body,
                source=stmt,
            )
        if isinstance(stmt, Do):
            iterator = stmt.iterator
            if isinstance(iterator, Variable):
                loop_var = lower_var(iterator, is_arg=False)
            else:
                loop_var = IRVar(name=str(iterator), source=iterator)
            return IRFor(
                var=loop_var,
                start=lower_expr(stmt.start),
                stop=lower_expr(stmt.end),
                step=lower_expr(stmt.step) if stmt.step is not None else None,
                body=[lower_stmt(s) for s in stmt.scope.get_statements()],
                source=stmt,
            )
        if isinstance(stmt, DoWhile):
            return IRWhile(
                cond=lower_expr(stmt.condition),
                body=[lower_stmt(s) for s in stmt.scope.get_statements()],
                source=stmt,
            )
        if isinstance(stmt, SelectCase):
            cases = [s for s in stmt.scope.get_statements() if isinstance(s, Case)]
            if not cases:
                return IROpaqueStmt(payload=stmt, source=stmt)

            def build_case(case_stmt):
                cond = IRBinary(
                    op="eq",
                    left=lower_expr(stmt.value),
                    right=lower_expr(case_stmt.value),
                )
                return IRIf(
                    cond=cond,
                    then=[lower_stmt(s) for s in case_stmt.scope.get_statements()],
                    else_=[],
                    source=case_stmt,
                )

            root = build_case(cases[0])
            current = root
            for case_stmt in cases[1:]:
                next_if = build_case(case_stmt)
                current.else_.append(next_if)
                current = next_if
            return root
        if isinstance(stmt, Return):
            return IRReturn(value=None, source=stmt)
        if isinstance(stmt, Allocate):
            return IRAllocate(
                var=lower_expr(stmt.target),
                dims=[lower_expr(dim) for dim in stmt.shape],
                source=stmt,
            )
        if isinstance(stmt, Deallocate):
            return IRDeallocate(var=lower_expr(stmt.array), source=stmt)
        return IROpaqueStmt(payload=stmt, source=stmt)

    args = [lower_var(var, is_arg=True) for var in subroutine.arguments.values()]
    locals_ = [lower_var(var, is_arg=False) for var in subroutine.get_local_variables().values()]
    body = [lower_stmt(stmt) for stmt in subroutine.scope.get_statements()]

    decl = subroutine.get_declaration()
    scope_ids = {id(stmt) for stmt in subroutine.scope.get_statements()}
    prelude_items: list[Any] = []
    for stmt in decl.get_statements():
        if id(stmt) in scope_ids:
            continue
        if isinstance(stmt, VariableDeclaration):
            continue
        prelude_items.append(stmt)

    return IRProcedure(
        name=subroutine.name,
        args=args,
        locals=locals_,
        body=body,
        result=None,
        source=subroutine,
        metadata={
            "syntax_subroutine": subroutine,
            "fortran_prelude_items": prelude_items,
            "fortran_pure": subroutine.pure,
            "fortran_elemental": subroutine.elemental,
            "fortran_bind_c": subroutine.bind_c,
        },
    )


def _lower_value_type(ftype, shape) -> IRValueType:
    dtype = _lower_type(ftype)
    if shape is SCALAR:
        return IRValueType(dtype=dtype, shape=None)
    if shape is UNKNOWN:
        return IRValueType(dtype=dtype, shape=IRShape(rank=None, dims=None, order="C"))
    dims = tuple(shape.dims)
    order = "F" if getattr(shape, "fortran_order", False) else "C"
    return IRValueType(dtype=dtype, shape=IRShape(rank=len(dims), dims=dims, order=order))


def _lower_type(ftype) -> IRType:
    name = getattr(ftype, "type", str(ftype))
    kind = None
    if hasattr(ftype, "get_kind_str"):
        kind = ftype.get_kind_str()
    return IRType(name=name, kind=kind)


def _safe_shape(expr):
    try:
        return expr._shape
    except NotImplementedError:
        return UNKNOWN


def _lower_indices(slice_) -> list[IRExpr | IRSlice]:
    if isinstance(slice_, tuple):
        return [_lower_single_index(item) for item in slice_]
    return [_lower_single_index(slice_)]


def _lower_single_index(item) -> IRExpr | IRSlice:
    if isinstance(item, slice):
        return _normalize_slice(item)
    expr = _lower_index_value(item) or IROpaqueExpr(payload=item, source=item)
    return _shift_expr(expr, -syntax_settings.array_lower_bound)


def _lower_index_value(value) -> IRExpr | None:
    if value is None:
        return None
    if isinstance(value, (int, float, bool, str)):
        return IRLiteral(value=value)
    if isinstance(value, LiteralNode):
        return IRLiteral(value=value.value)
    if isinstance(value, Variable):
        return IRVarRef(var=IRVar(name=value.name), source=value)
    if isinstance(value, BinaryOperationNode):
        op = _map_binary_op(value.op)
        left = _lower_index_value(value.left) or IROpaqueExpr(payload=value.left, source=value.left)
        right = _lower_index_value(value.right) or IROpaqueExpr(
            payload=value.right, source=value.right
        )
        return IRBinary(op=op, left=left, right=right)
    if isinstance(value, GetItem):
        base = _lower_index_value(value.variable) or IROpaqueExpr(
            payload=value.variable, source=value.variable
        )
        return IRGetItem(base=base, indices=_lower_indices(value.sliced))
    if isinstance(value, GetAttr):
        base = _lower_index_value(value.variable) or IROpaqueExpr(
            payload=value.variable, source=value.variable
        )
        return IRGetAttr(base=base, name=value.attr)
    if isinstance(value, FunctionCall):
        callee = _lower_index_value(value.function) or IROpaqueExpr(
            payload=value.function, source=value.function
        )
        return IRCallExpr(
            callee=callee,
            args=[
                arg for arg in (_lower_index_value(a) for a in value.arguments) if arg is not None
            ],
        )
    if isinstance(value, IntrinsicFunction):
        return IRIntrinsic(
            name=getattr(value, "token", ""),
            args=[
                arg for arg in (_lower_index_value(a) for a in value.arguments) if arg is not None
            ],
        )
    return IROpaqueExpr(payload=value, source=value)


def _normalize_slice(slice_: slice) -> IRSlice:
    lbound = syntax_settings.array_lower_bound
    c_like = syntax_settings.c_like_bounds

    if slice_.start is None:
        start = IRLiteral(value=0)
    else:
        start = _lower_index_value(slice_.start) or IROpaqueExpr(
            payload=slice_.start, source=slice_.start
        )
        start = _shift_expr(start, -lbound)

    stop = None
    if slice_.stop is not None:
        stop_expr = _lower_index_value(slice_.stop) or IROpaqueExpr(
            payload=slice_.stop, source=slice_.stop
        )
        shift = (0 if c_like else 1) - lbound
        stop = _shift_expr(stop_expr, shift)

    step = None
    if slice_.step is not None:
        step = _lower_index_value(slice_.step) or IROpaqueExpr(
            payload=slice_.step, source=slice_.step
        )

    return IRSlice(start=start, stop=stop, step=step)


def _shift_expr(expr: IRExpr, delta: int) -> IRExpr:
    if delta == 0:
        return expr
    if isinstance(expr, IRLiteral) and isinstance(expr.value, (int, float)):
        return IRLiteral(value=expr.value + delta)
    op = "add" if delta > 0 else "sub"
    return IRBinary(op=op, left=expr, right=IRLiteral(value=abs(delta)))


def _map_binary_op(op: str) -> str:
    return _BINARY_OPS.get(op, op)
