import numpy as np
import subprocess as sp
from pathlib import Path
import tempfile
import importlib.util
import sys
import sysconfig
import pickle
import shutil
from dataclasses import dataclass
from typing import Any
import textwrap
import inspect

from numeta.external_library import ExternalLibrary

from .settings import settings
from .builder_helper import BuilderHelper
from .syntax import Subroutine, Variable
from .syntax.expressions import ExpressionNode, GetAttr, GetItem
from .datatype import DataType, ArrayType, get_datatype, size_t
from .capi_interface import CAPIInterface
from .types_hint import comptime
from .array_shape import ArrayShape, SCALAR, UNKNOWN


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


class NumetaCompilationTarget(ExternalLibrary):
    """
    To link NumetaFunctions when called by other NumetaFunctions
    """

    _registry: list["NumetaCompilationTarget"] = []

    @classmethod
    def registered_functions(cls) -> list["NumetaCompilationTarget"]:
        """Return a snapshot of all instantiated :class:`NumetaFunction` objects."""
        return list(cls._registry)

    @classmethod
    def clear_registered_functions(cls) -> None:
        """Clear the registry of instantiated :class:`NumetaFunction` objects."""
        print("clearing")
        cls._registry.clear()

    def __init__(
        self,
        name,
        symbolic_function,
        directory: Path,
        *,
        do_checks,
        compile_flags,
    ):
        super().__init__(name, to_link=False)
        self.symbolic_function = symbolic_function
        self.directory = directory
        self.do_checks = do_checks
        self.compile_flags = compile_flags

        self._obj_files = None
        self.compiled_with_capi_file = None
        self.capi_name = None

        # Register this instance for later inspection via the class registry
        type(self)._registry.append(self)

        # Object files of all the NumetaCompilationTarget needed by this one
        self._nested_obj_files = None

    @property
    def obj_files(self):
        if self._obj_files is None:
            self._obj_files = self.compile_fortran()
        return self._obj_files

    def copy(self, directory):
        result = NumetaCompilationTarget(
            self.name,
            self.symbolic_function,
            directory,
            do_checks=self.do_checks,
            compile_flags=self.compile_flags,
        )

        new_dir = directory / self.name
        new_dir.mkdir(exist_ok=True)
        if self._obj_files is not None:
            new_obj_file = new_dir / self.obj_files[0].name
            shutil.copy(self.obj_files[0], new_obj_file)
            result._obj_files = [new_obj_file]
        if self.compiled_with_capi_file is not None:
            new_lib_file = new_dir / self.compiled_with_capi_file.name
            shutil.copy(self.compiled_with_capi_file, new_lib_file)
            result.compiled_with_capi_file = new_lib_file
            result.capi_name = self.capi_name
        if len(self.get_nested_obj_files()) != 0:
            raise NotImplementedError("Dumping of nested numeta calls not implemented yet")

        return result

    def get_nested_obj_files(self):
        obj_files = set()
        queue = self.symbolic_function.get_dependencies().values()
        while queue:
            new_queue = []
            for lib in queue:
                if isinstance(lib, NumetaCompilationTarget):
                    new_obj = lib.obj_files[0]
                    if new_obj not in obj_files:
                        obj_files.add(new_obj)
                        new_queue += lib.symbolic_function.get_dependencies().values()
            queue = new_queue
        return obj_files

    def _run_command(self, command, cwd):
        sp_run = sp.run(
            command,
            cwd=cwd,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
        )
        if sp_run.returncode != 0:
            error_message = "Error while compiling, the command was:\n"
            error_message += " ".join(command) + "\n"
            error_message += "The output was:\n"
            error_message += textwrap.indent(sp_run.stdout.decode("utf-8"), "    ")
            error_message += textwrap.indent(sp_run.stderr.decode("utf-8"), "    ")
            raise Warning(error_message)
        return sp_run

    def compile_fortran(self):
        """
        Compile Fortran source files using gfortran and return the resulting object file.
        """

        fortran_src = self.directory / f"{self.name}_src.f90"
        fortran_src.write_text(self.symbolic_function.get_code())

        output = self.directory / f"{self.name}_fortran.o"

        include_dirs = []
        additional_flags = []
        dependencies = self.symbolic_function.get_dependencies().values()

        for lib in dependencies:

            if lib.include is not None:
                include_dirs.append(lib.include)
            if lib.additional_flags is not None:
                if isinstance(lib.additional_flags, str):
                    additional_flags.extend(lib.additional_flags.split())
                else:
                    additional_flags.append(lib.additional_flags)

        command = ["gfortran"]
        command.extend(["-fopenmp"])
        command.extend(self.compile_flags)
        command.extend(["-fPIC", "-c", "-o", str(output)])
        command.append(str(fortran_src))
        command.extend([f"-I{inc_dir}" for inc_dir in include_dirs])
        command.extend(additional_flags)

        self._run_command(command, cwd=self.directory)

        return [output]

    def compile_with_capi_interface(
        self,
        capi_name,
        capi_obj,
    ):
        """
        Compiles Fortran code and constructs a C API interface,
        then compiles them into a shared library and loads the module.

        Parameters:
            *args: Arguments to pass to compile_fortran_function.

        Returns:
            tuple: (compiled function, subroutine)
        """
        self.capi_name = capi_name

        local_dir = self.directory / self.name
        local_dir.mkdir(exist_ok=True)

        self.compiled_with_capi_file = local_dir / f"lib{self.name}_module.so"

        libraries = [
            "gfortran",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
        ]
        libraries_dirs = []
        include_dirs = [sysconfig.get_paths()["include"], np.get_include()]
        extra_objects = self.get_nested_obj_files()
        additional_flags = []

        for lib in self.symbolic_function.get_dependencies().values():

            if lib.obj_files is not None:
                extra_objects |= set(lib.obj_files)

            if lib.to_link:
                libraries.append(lib.name)
                if lib.directory is not None:
                    libraries_dirs.append(lib.directory)
                if lib.include is not None:
                    include_dirs.append(lib.include)
                if lib.additional_flags is not None:
                    if isinstance(lib.additional_flags, str):
                        additional_flags.extend(lib.additional_flags.split())
                    else:
                        additional_flags.append(lib.additional_flags)

        command = ["gcc"]
        command.extend(self.compile_flags)
        command.extend(["-fopenmp"])
        command.extend(["-fPIC", "-shared", "-o", str(self.compiled_with_capi_file)])
        command.extend([str(*self.obj_files), str(capi_obj)])
        command.extend([str(obj) for obj in extra_objects])
        command.extend([f"-l{lib}" for lib in libraries])
        command.extend([f"-L{lib_dir}" for lib_dir in libraries_dirs])
        command.extend([f"-I{inc_dir}" for inc_dir in include_dirs])
        command.extend(additional_flags)

        sp_run = sp.run(
            command,
            cwd=local_dir,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
        )
        if sp_run.returncode != 0:
            error_message = "Error while compiling, the command was:\n"
            error_message += " ".join(command) + "\n"
            error_message += "The output was:\n"
            error_message += textwrap.indent(sp_run.stdout.decode("utf-8"), "    ")
            error_message += textwrap.indent(sp_run.stderr.decode("utf-8"), "    ")
            raise Warning(error_message)

    def load_with_capi(self):
        if self.capi_name is None:
            raise ValueError("Function should be compiled before loading it")
        spec = importlib.util.spec_from_file_location(self.capi_name, self.compiled_with_capi_file)
        compiled_sub = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compiled_sub)
        return getattr(compiled_sub, self.name)


class NumetaFunction:
    """
    Representation of a JIT-compiled function.
    """

    def __init__(
        self,
        func,
        directory=None,
        do_checks=True,
        compile_flags="-O3 -march=native",
        namer=None,
        inline: bool | int = False,
    ) -> None:
        super().__init__()
        self.name = func.__name__
        if directory is None:
            directory = tempfile.mkdtemp()
        self.directory = Path(directory).absolute()
        self.directory.mkdir(exist_ok=True)
        self.do_checks = do_checks
        self.compile_flags = compile_flags.split()

        self.namer = namer
        self.inline = inline
        self._func = func

        # To store the dependencies of the compiled functions to other numeta generated functions.
        self._py_signature = inspect.signature(func)

        self.params = []
        self.catch_var_positional_name = "args"

        for name, parameter in self._py_signature.parameters.items():
            is_comp = func.__annotations__.get(name) is comptime
            pinfo = ParameterInfo(name, parameter.kind, parameter.default, is_comp)
            self.params.append(pinfo)

            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                self.catch_var_positional_name = name

        self.fixed_param_indices = [
            i
            for i, p in enumerate(self.params)
            if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        self.n_positional_or_default_args = len(self.fixed_param_indices)

        # Variables to populate
        self._signature_to_name = {}
        self.return_signatures = {}  # Only needed if i create symbolic and after compile
        self._compiled_targets = {}
        self._fast_call = {}

    def dump(self, directory):
        """
        Dumps the compiled function to a file.
        """
        directory = Path(directory)
        directory.mkdir(exist_ok=True)

        # Copy the libraries to the new directory
        new_compiled_target = {}
        for signature, compiled_target in self._compiled_targets.items():
            new_compiled_target[signature] = compiled_target.copy(directory)

        with open(directory / f"{self.name}.pkl", "wb") as f:
            pickle.dump(self._signature_to_name, f)
            pickle.dump(self.return_signatures, f)
            pickle.dump(new_compiled_target, f)

    def load(self, directory):
        """
        Loads the compiled function from a file.
        """
        self._fast_call = {}
        with open(Path(directory) / f"{self.name}.pkl", "rb") as f:
            self._signature_to_name = pickle.load(f)
            self._return_signatures = pickle.load(f)
            self._compiled_targets = pickle.load(f)
        self.return_signatures = {}

    def get_name(self, signature):
        if signature not in self._signature_to_name:
            if self.namer is None:
                name = f"{self.name}_{len(self._signature_to_name)}"
            else:
                name = self.namer(*signature)
            self._signature_to_name[signature] = name
        return self._signature_to_name[signature]

    def get_signature_idx(self, func):
        return

    def get_symbolic_functions(self):
        return [v.symbolic_function for v in self._compiled_targets.values()]

    def run_symbolic(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def _get_signature_and_runtime_args_from_args(self, args, kwargs):
        """
        This method quickly extracts the signature and runtime arguments from the provided args.
        If the runtime arguments are not all numpy arrays or numeric types, the call is not to be executed.
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
                    arg_signature = (
                        name,
                        dtype.get_numpy(),
                        arg._shape.rank,
                        arg._shape.fortran_order,
                        intent,
                        arg._shape.dims,
                    )
            else:
                raise ValueError(f"Argument {name} of type {type(arg)} is not supported")

            return arg_signature

        runtime_args = []
        signature = [None] * self.n_positional_or_default_args

        unused_kwargs = kwargs
        pos_idx = 0

        for fi, param_idx in enumerate(self.fixed_param_indices):
            param = self.params[param_idx]

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
                name = (self.catch_var_positional_name, j)
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

    def get_signature(self, *args, **kwargs):
        _, signature, _ = self._get_signature_and_runtime_args_from_args(args, kwargs)
        return signature

    def _convert_signature_to_argument_specs(self, signature):
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

            if i < self.n_positional_or_default_args:
                param = self.params[self.fixed_param_indices[i]]
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

    def __call__(self, *args, **kwargs):
        """
        **Note**: Support for fixed shapes for arrays works when not using numpy arrays.
        But if the function is called with numeta types hint like nm.float32[2, 3] it will create a symbolic function with fixed shapes arguments.
        Calling it is another story and not implemented yet.
        """

        if BuilderHelper.current_builder is not None:
            # We are already contructing a symbolic function

            _, signature, runtime_args = self._get_signature_and_runtime_args_from_args(
                args, kwargs
            )

            symbolic_fun = self.get_symbolic_function(signature)

            # first check the runtime arguments
            # they should be symbolic nodes
            from .syntax.tools import check_node

            runtime_args = [check_node(arg) for arg in runtime_args]

            # Optionally add the array descriptor for arrays with runtime-dependent dimensions
            full_runtime_args = []
            for arg in runtime_args:
                if settings.add_shape_descriptors and arg._shape.has_comptime_undefined_dims():
                    full_runtime_args.append(arg._get_shape_descriptor())
                full_runtime_args.append(arg)

            do_inline = False
            if isinstance(self.inline, bool):
                do_inline = self.inline
            elif isinstance(self.inline, int):
                if symbolic_fun.count_statements() <= self.inline:
                    do_inline = True

            if do_inline:
                from .syntax.inline import inline as inline_call

                inline_call(symbolic_fun, *full_runtime_args)
            else:
                # This add a Call statement to the current builder
                symbolic_fun(*full_runtime_args)
        else:

            # TODO: probably overhead, to do in C?
            to_execute, signature, runtime_args = self._get_signature_and_runtime_args_from_args(
                args, kwargs
            )

            if not to_execute:
                return self.get_symbolic_function(signature)

            return self.execute_function(signature, runtime_args)

    def get_symbolic_function(self, signature):
        if signature not in self._compiled_targets:
            self.construct_symbolic_function(signature)
        return self._compiled_targets[signature].symbolic_function

    def construct_symbolic_function(self, signature):
        name = self.get_name(signature)
        argument_specs = self._convert_signature_to_argument_specs(signature)

        sub = Subroutine(name)
        builder = BuilderHelper(self, sub, signature)

        def convert_argument_spec_to_variable(arg_spec):
            """
            Converts an ArgumentSpec to a Variable.
            """
            ftype = arg_spec.datatype.get_fortran()
            if arg_spec.rank == 0:
                return Variable(arg_spec.name, ftype=ftype, shape=SCALAR, intent=arg_spec.intent)
            elif arg_spec.shape is UNKNOWN:
                return Variable(
                    arg_spec.name,
                    ftype=ftype,
                    shape=UNKNOWN,
                    intent=arg_spec.intent,
                )
            elif arg_spec.shape.has_comptime_undefined_dims():
                if settings.add_shape_descriptors:
                    # The shape will to be passed as a separate argument
                    dim_var = Variable(
                        f"shape_{arg_spec.name}",
                        ftype=size_t.get_fortran(bind_c=True),
                        shape=ArrayShape((arg_spec.rank,)),
                        intent="in",
                    )
                    sub.add_variable(dim_var)

                    shape = ArrayShape(
                        tuple([dim_var[i] for i in range(arg_spec.rank)]),
                        fortran_order=arg_spec.shape.fortran_order,
                    )
                else:
                    shape = UNKNOWN
                return Variable(
                    arg_spec.name,
                    ftype=ftype,
                    shape=shape,
                    intent=arg_spec.intent,
                )
            else:
                # The dimension is fixed
                return Variable(
                    arg_spec.name,
                    ftype=ftype,
                    shape=arg_spec.shape,
                    intent=arg_spec.intent,
                )

        symbolic_args = []
        symbolic_kwargs = {}
        for arg in argument_specs:
            if arg.is_comptime:
                if arg.is_keyword:
                    symbolic_kwargs[arg.name] = arg.comptime_value
                else:
                    symbolic_args.append(arg.comptime_value)
            else:
                var = convert_argument_spec_to_variable(arg)
                # Add the variable to the subroutine
                sub.add_variable(var)
                if arg.is_keyword:
                    symbolic_kwargs[arg.name] = var
                else:
                    symbolic_args.append(var)

        return_signature = builder.build(*symbolic_args, **symbolic_kwargs)
        self.return_signatures[signature] = return_signature

        self._compiled_targets[signature] = NumetaCompilationTarget(
            self.get_name(signature),
            sub,
            directory=self.directory,
            do_checks=self.do_checks,
            compile_flags=self.compile_flags,
        )

        sub.parent = self._compiled_targets[signature]

    def compile_function(self, signature):
        if signature not in self._compiled_targets:
            self.construct_symbolic_function(signature)

        capi_name = f"{self.name}_capi"
        capi_interface = CAPIInterface(
            self.get_name(signature),
            module_name=capi_name,
            args_details=self._convert_signature_to_argument_specs(signature),
            return_specs=self.return_signatures[signature],
            directory=self.directory,
            compile_flags=self.compile_flags,
            do_checks=self.do_checks,
        )
        capi_obj = capi_interface.generate()

        self._compiled_targets[signature].compile_with_capi_interface(capi_name, capi_obj)

    def load_function(self, signature):
        if signature not in self._compiled_targets:
            self.compile_function(signature)
        self._fast_call[signature] = self._compiled_targets[signature].load_with_capi()

    def execute_function(self, signature, runtime_args):
        if signature not in self._fast_call:
            self.load_function(signature)
        return self._fast_call[signature](*runtime_args)
