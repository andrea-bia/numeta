import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np

from .array_shape import ArrayShape, SCALAR, UNKNOWN
from .datatype import DataType, ArrayType, get_datatype
from .settings import settings
from .syntax import Variable
from .syntax.expressions import ExpressionNode, GetAttr, GetItem
from .types_hint import comptime


@dataclass(frozen=True)
class ArgumentSpec:
    """
    This class is used to store the details of the arguments of the function.
    The ones that are compile-time are stored in the is_comptime attribute.
    """

    name: str
    is_comptime: bool = False
    comptime_value: Any = None
    datatype: DataType | None = None
    shape: ArrayShape | None = None
    rank: int = 0  # rank of the array, 0 for scalar
    intent: str = "inout"  # can be "in" or "inout"
    to_pass_by_value: bool = False
    is_keyword: bool = False


@dataclass(frozen=True)
class ParameterInfo:
    """Store metadata about a Python function parameter."""

    name: str
    kind: inspect._ParameterKind
    default: Any = inspect._empty
    is_comptime: bool = False


def parse_function_parameters(func):
    py_signature = inspect.signature(func)

    params = []
    catch_var_positional_name = "args"

    for name, parameter in py_signature.parameters.items():
        is_comp = func.__annotations__.get(name) is comptime
        params.append(ParameterInfo(name, parameter.kind, parameter.default, is_comp))

        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            catch_var_positional_name = name

    fixed_param_indices = [
        i
        for i, p in enumerate(params)
        if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    n_positional_or_default_args = len(fixed_param_indices)

    return params, fixed_param_indices, n_positional_or_default_args, catch_var_positional_name


def get_signature_and_runtime_args(
    args,
    kwargs,
    *,
    params,
    fixed_param_indices,
    n_positional_or_default_args,
    catch_var_positional_name,
):
    """
    This method quickly extracts the signature and runtime arguments from the provided args.
    If the runtime arguments are not all numpy arrays or numeric types, the call is not to be
    executed.
    It returns a tuple of:
    - to_execute: a boolean indicating if the function can be executed with the provided args
    - signature: a tuple of the signature of the function
    - runtime_args: a list of runtime arguments to be passed to run the function
    A signature is a tuple of tuples, where each inner tuple represents an argument.
    - (name, dtype,) is scalar types passed by value
    - (name, dtype, 0) is for scalar types passed by reference
    - (name, dtype, rank) for numpy arrays
    - (name, dtype, rank, has_fortran_order) to set the Fortran order
    - (name, dtype, rank, has_fortran_order, intent) intent can be "in" or "inout"
    - (name, dtype, rank, has_fortran_order, intent, shape) if the shape is know at comptime
    **name** is a tuple is the argument comes from a variable positional argument (*args)
    """

    to_execute = True

    def get_signature_from_arg(arg, name):
        nonlocal to_execute

        if isinstance(arg, np.ndarray):
            arg_signature = (name, arg.dtype, len(arg.shape), np.isfortran(arg))
        elif isinstance(arg, (int, float, complex)):
            arg_signature = (
                name,
                type(arg),
            )
        elif isinstance(arg, np.generic):
            # it is a numpy scalar like np.int32(1) or np.float64(1.0) or a struct
            # A struct is mutable
            if arg.dtype.names is not None:
                arg_signature = (name, arg.dtype, 0)
            else:
                arg_signature = (
                    name,
                    arg.dtype,
                )
        elif isinstance(arg, ArrayType):
            to_execute = False
            if arg.shape is UNKNOWN or (
                not settings.add_shape_descriptors and arg.shape.has_comptime_undefined_dims()
            ):
                # it is a pointer
                arg_signature = (name, arg.dtype.get_numpy(), None, arg.shape.fortran_order)
            elif arg.shape.has_comptime_undefined_dims():
                arg_signature = (
                    name,
                    arg.dtype.get_numpy(),
                    arg.shape.rank,
                    arg.shape.fortran_order,
                )
            else:
                arg_signature = (
                    name,
                    arg.dtype.get_numpy(),
                    arg.shape.rank,
                    arg.shape.fortran_order,
                    "inout",
                    arg.shape.dims,
                )
        elif isinstance(arg, type) and issubclass(arg, DataType):
            to_execute = False
            arg_signature = (
                name,
                arg.get_numpy(),
            )
        elif isinstance(arg, ExpressionNode):
            to_execute = False
            ftype = arg._ftype
            dtype = DataType.from_ftype(ftype)
            # Let's stay safe, let's assume is an expression so intent is in
            intent = "in"
            # These are the cases where we can assume it is an inout argument
            # becase the intent can be only "in" or "inout"
            if isinstance(arg, Variable) and arg.intent != "in":
                intent = "inout"
            if isinstance(arg, GetAttr) and arg.variable.intent != "in":
                intent = "inout"
            if isinstance(arg, GetItem) and arg.variable.intent != "in":
                intent = "inout"

            if arg._shape is SCALAR:
                if intent == "inout":
                    arg_signature = (name, dtype.get_numpy(), 0, False, intent)
                else:
                    arg_signature = (
                        name,
                        dtype.get_numpy(),
                    )
            elif arg._shape is UNKNOWN or (
                not settings.add_shape_descriptors and arg._shape.has_comptime_undefined_dims()
            ):
                arg_signature = (name, dtype.get_numpy(), None, False, intent)
            elif arg._shape.has_comptime_undefined_dims():
                arg_signature = (
                    name,
                    dtype.get_numpy(),
                    arg._shape.rank,
                    arg._shape.fortran_order,
                    intent,
                )
            else:
                if not settings.ignore_fixed_shape_in_nested_calls:
                    arg_signature = (
                        name,
                        dtype.get_numpy(),
                        arg._shape.rank,
                        arg._shape.fortran_order,
                        intent,
                        arg._shape.dims,
                    )
                else:
                    arg_signature = (name, dtype.get_numpy(), None, False, intent)
        else:
            raise ValueError(f"Argument {name} of type {type(arg)} is not supported")

        return arg_signature

    runtime_args = []
    signature = [None] * n_positional_or_default_args

    unused_kwargs = kwargs
    pos_idx = 0

    for fi, param_idx in enumerate(fixed_param_indices):
        param = params[param_idx]

        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            if pos_idx < len(args):
                arg = args[pos_idx]
                pos_idx += 1
            elif param.name in unused_kwargs:
                arg = unused_kwargs.pop(param.name)
            elif param.default is not inspect._empty:
                arg = param.default
            else:
                raise ValueError(f"Missing required argument: {param.name}")
        else:  # KEYWORD_ONLY
            if param.name in unused_kwargs:
                arg = unused_kwargs.pop(param.name)
            elif param.default is not inspect._empty:
                arg = param.default
            else:
                raise ValueError(f"Missing required argument: {param.name}")

        if param.is_comptime:
            signature[fi] = arg
        else:
            signature[fi] = get_signature_from_arg(arg, param.name)
            runtime_args.append(arg)

    # catch the *args variable positional arguments
    if pos_idx < len(args):
        for j, arg in enumerate(args[pos_idx:]):
            name = (catch_var_positional_name, j)
            signature.append(get_signature_from_arg(arg, name))
            runtime_args.append(arg)

    # catch the **kwargs variable keyword arguments
    unused_kwargs_keys = (
        unused_kwargs.keys() if not settings.reorder_kwargs else sorted(unused_kwargs.keys())
    )
    for name in unused_kwargs_keys:
        arg = unused_kwargs[name]
        signature.append(get_signature_from_arg(arg, name))
        runtime_args.append(arg)

    return to_execute, tuple(signature), runtime_args


def convert_signature_to_argument_specs(
    signature,
    *,
    params,
    fixed_param_indices,
    n_positional_or_default_args,
):
    """
    Converts a signature tuple into a list of ArgumentSpec objects.
    A signature is a tuple of tuples, where each inner tuple represents an argument.
    """

    def convert_arg_to_argument_spec(arg, is_keyword, name=None):
        name = arg[0] if name is None else name

        dtype = get_datatype(arg[1])
        if len(arg) == 2:
            # it is a numeric type or a string
            # So the intent will be always "in"
            # but complex numbers cannot be passed by value because of C
            ap = ArgumentSpec(
                name,
                datatype=dtype,
                shape=SCALAR,
                to_pass_by_value=dtype.can_be_value(),
                intent="in",
                is_keyword=is_keyword,
            )
        else:
            # for numpy arrays arg[1] is the rank, for the other types it is the shape
            rank = arg[2]

            fortran_order = False
            if len(arg) == 4:
                fortran_order = arg[3]

            intent = "inout"
            if len(arg) == 5:
                intent = arg[4]

            if rank is None:
                shape = UNKNOWN
            elif rank == 0:
                shape = SCALAR
            else:
                shape = ArrayShape([None] * rank, fortran_order=fortran_order)

            if len(arg) == 6:
                # it means that the shape is known at comptime
                shape = ArrayShape(arg[5], fortran_order=fortran_order)

            ap = ArgumentSpec(
                name,
                datatype=dtype,
                rank=rank,
                shape=shape,
                intent=intent,
                is_keyword=is_keyword,
            )

        return ap

    signature_spec = []
    for i, arg in enumerate(signature):

        if i < n_positional_or_default_args:
            param = params[fixed_param_indices[i]]
            if param.is_comptime:
                ap = ArgumentSpec(
                    param.name,
                    is_comptime=True,
                    comptime_value=arg,
                    is_keyword=param.kind == inspect.Parameter.KEYWORD_ONLY,
                )
            else:
                is_keyword = param.kind == inspect.Parameter.KEYWORD_ONLY
                ap = convert_arg_to_argument_spec(arg, is_keyword)
        elif isinstance(arg[0], tuple):
            # it is a positional argument called with *args
            is_keyword = False
            name = f"{arg[0][0]}_{arg[0][1]}"
            ap = convert_arg_to_argument_spec(arg, is_keyword, name=name)
        else:
            is_keyword = True
            ap = convert_arg_to_argument_spec(arg, is_keyword)

        signature_spec.append(ap)

    return signature_spec
