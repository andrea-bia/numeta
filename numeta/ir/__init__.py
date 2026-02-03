from .nodes import (
    IRNode,
    IRType,
    IRShape,
    IRValueType,
    IRVar,
    IRStruct,
    IRModule,
    IRProcedure,
    IRStmt,
    IRExpr,
    IRLiteral,
    IRVarRef,
    IRBinary,
    IRUnary,
    IRCall,
    IRCallExpr,
    IRGetItem,
    IRGetAttr,
    IRIntrinsic,
    IRSlice,
    IRAssign,
    IRIf,
    IRFor,
    IRWhile,
    IRReturn,
    IRAllocate,
    IRDeallocate,
    IROpaqueExpr,
    IROpaqueStmt,
)
from .lowering import lower_subroutine
from typing import Any

__all__ = [
    "IRNode",
    "IRType",
    "IRShape",
    "IRValueType",
    "IRVar",
    "IRStruct",
    "IRModule",
    "IRProcedure",
    "IRStmt",
    "IRExpr",
    "IRLiteral",
    "IRVarRef",
    "IRBinary",
    "IRUnary",
    "IRCall",
    "IRCallExpr",
    "IRGetItem",
    "IRGetAttr",
    "IRIntrinsic",
    "IRSlice",
    "IRAssign",
    "IRIf",
    "IRFor",
    "IRWhile",
    "IRReturn",
    "IRAllocate",
    "IRDeallocate",
    "IROpaqueExpr",
    "IROpaqueStmt",
    "lower_subroutine",
    "FortranEmitter",
    "CEmitter",
]


def __getattr__(name: str) -> Any:
    if name == "FortranEmitter":
        from numeta.fortran.emitter import FortranEmitter

        return FortranEmitter
    if name == "CEmitter":
        from numeta.c.emitter import CEmitter

        return CEmitter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
