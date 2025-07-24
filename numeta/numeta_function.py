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


class NumetaFunction:
    def __init__(
        self,
        func,
        directory=None,
        do_checks=True,
        compile_flags="-O3 -march=native",
        namer=None,
        inline: bool | int = False,
    ) -> None:
        self.name = func.__name__
        if directory is None:
            directory = tempfile.mkdtemp()
        self.directory = Path(directory).absolute()
        self.directory.mkdir(exist_ok=True)
        self.do_checks = do_checks
        self.compile_flags = compile_flags.split()
        self.namer = namer
        self.inline = inline

        self.__func = func
        self.__signature_to_name = {}
        self.__symbolic_functions = {}
        self.__libraries = {}
        self.__loaded_functions = {}
        self.return_signatures = {}

        # To store the dependencies of the compiled functions to other numeta generated functions.
        self.__dependencies = {}

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

    def dump(self, directory):
        """
        Dumps the compiled function to a file.
        """
        directory = Path(directory)
        directory.mkdir(exist_ok=True)

        # Copy the libraries to the new directory
        new_libraries = {}
        for signature, (lib_name, lib_obj, lib_file) in self.__libraries.items():
            name = self.get_name(signature)
            new_directory = directory / name
            new_directory.mkdir(exist_ok=True)
            new_lib_obj = new_directory / lib_obj.name
            new_lib_file = new_directory / lib_file.name
            shutil.copy(lib_obj, new_lib_obj)
            shutil.copy(lib_file, new_lib_file)
            new_libraries[signature] = (lib_name, new_lib_obj, new_lib_file)

        with open(directory / f"{self.name}.pkl", "wb") as f:
            pickle.dump(self.__signature_to_name, f)
            pickle.dump(self.__symbolic_functions, f)
            pickle.dump(new_libraries, f)

        if self.__dependencies:
            raise NotImplementedError("Dependencies are not yet supported in dump method")

    def load(self, directory):
        """
        Loads the compiled function from a file.
        """
        self.__loaded_functions = {}
        with open(Path(directory) / f"{self.name}.pkl", "rb") as f:
            self.__signature_to_name = pickle.load(f)
            self.__symbolic_functions = pickle.load(f)
            self.__libraries = pickle.load(f)

    def get_name(self, signature):
        if signature not in self.__signature_to_name:
            if self.namer is None:
                name = f"{self.name}_{len(self.__signature_to_name)}"
            else:
                name = self.namer(signature)
            self.__signature_to_name[signature] = name
        return self.__signature_to_name[signature]

    def get_signature_idx(self, func):
        return

    def get_symbolic_functions(self):
        return list(self.__symbolic_functions.values())

    def run_symbolic(self, *args, **kwargs):
        return self.__func(*args, **kwargs)

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
                if arg.shape is UNKNOWN:
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
                elif arg._shape is UNKNOWN:
                    arg_signature = (name, dtype.get_numpy(), 0, False, intent)
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

        unused_kwargs = dict(kwargs)
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
        for name in sorted(unused_kwargs.keys()):
            arg = unused_kwargs[name]
            signature.append(get_signature_from_arg(arg, name))
            runtime_args.append(arg)

        return to_execute, tuple(signature), runtime_args

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

            # Should add the array descriptor for each array
            full_runtime_args = []
            for arg in runtime_args:
                if arg._shape.has_comptime_undefined_dims():
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

                # Add nested dependencies
                caller = BuilderHelper.get_current_builder().numeta_function
                caller_identifier = BuilderHelper.get_current_builder().signature
                for dep, dep_signature in self.__dependencies.get(signature, {}).values():
                    if caller_identifier not in caller.__dependencies:
                        caller.__dependencies[caller_identifier] = {}
                    caller.__dependencies[caller_identifier][dep.name, dep_signature] = (
                        dep,
                        dep_signature,
                    )
            else:
                # This add a Call statement to the current builder
                symbolic_fun(*full_runtime_args)

                # Add this function to the dependencies of the current builder
                caller = BuilderHelper.get_current_builder().numeta_function
                caller_identifier = BuilderHelper.get_current_builder().signature
                if caller_identifier not in caller.__dependencies:
                    caller.__dependencies[caller_identifier] = {}
                caller.__dependencies[caller_identifier][self.name, signature] = (
                    self,
                    signature,
                )
                # Add nested dependencies
                for dep, dep_signature in self.__dependencies.get(signature, {}).values():
                    caller.__dependencies[caller_identifier][dep.name, dep_signature] = (
                        dep,
                        dep_signature,
                    )

        else:

            # TODO: probably overhead, to do in C?
            to_execute, signature, runtime_args = self._get_signature_and_runtime_args_from_args(
                args, kwargs
            )

            if not to_execute:
                return self.get_symbolic_function(signature)

            return self.execute_function(signature, runtime_args)

    def get_symbolic_function(self, signature):
        if signature not in self.__symbolic_functions:
            self.construct_symbolic_function(signature)
        return self.__symbolic_functions[signature]

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
                return Variable(arg_spec.name, ftype=ftype, shape=SCALAR, intent=arg.intent)
            elif arg_spec.shape is UNKNOWN:
                return Variable(
                    arg_spec.name,
                    ftype=ftype,
                    shape=UNKNOWN,
                    intent=arg.intent,
                )
            elif arg_spec.shape.has_comptime_undefined_dims():
                # The shape will to be passed as a separate argument
                dim_var = builder.generate_local_variables(
                    f"fc_n",
                    ftype=size_t.get_fortran(bind_c=True),
                    shape=ArrayShape((arg_spec.rank,)),
                    intent="in",
                )
                sub.add_variable(dim_var)

                shape = ArrayShape(
                    tuple([dim_var[i] for i in range(arg_spec.rank)]),
                    fortran_order=arg_spec.shape.fortran_order,
                )
                return Variable(
                    arg_spec.name,
                    ftype=ftype,
                    shape=shape,
                    intent=arg.intent,
                )
            else:
                # The dimension is fixed
                return Variable(
                    arg_spec.name,
                    ftype=ftype,
                    shape=arg_spec.shape,
                    intent=arg.intent,
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
        self.__symbolic_functions[signature] = sub
        self.return_signatures[signature] = return_signature

    def compile_function(
        self,
        signature,
    ):
        """
        Compiles Fortran code and constructs a C API interface,
        then compiles them into a shared library and loads the module.

        Parameters:
            *args: Arguments to pass to compile_fortran_function.

        Returns:
            tuple: (compiled function, subroutine)
        """
        name = self.get_name(signature)

        local_dir = self.directory / name
        local_dir.mkdir(exist_ok=True)

        symbolic_function = self.get_symbolic_function(signature)
        fortran_obj = self.compile_fortran(name, symbolic_function, local_dir)

        capi_name = f"{name}_capi"
        argument_specs = self._convert_signature_to_argument_specs(signature)
        return_specs = self.return_signatures[signature]
        capi_interface = CAPIInterface(
            name,
            capi_name,
            argument_specs,
            return_specs,
            local_dir,
            self.compile_flags,
            self.do_checks,
        )
        capi_obj = capi_interface.generate()

        compiled_library_file = local_dir / f"lib{name}_module.so"

        libraries = [
            "gfortran",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
        ]
        libraries_dirs = []
        include_dirs = [sysconfig.get_paths()["include"], np.get_include()]
        extra_objects = []
        additional_flags = []

        if signature in self.__dependencies:
            for dep, dep_signature in self.__dependencies[signature].values():
                if dep_signature not in dep.__libraries:
                    dep.compile_function(dep_signature)
                _, dep_obj, _ = dep.__libraries[dep_signature]
                extra_objects.append(dep_obj)

        for external_dep in symbolic_function.get_external_dependencies().values():
            lib = None
            if hasattr(external_dep, "library"):
                if external_dep.library is not None:
                    lib = external_dep.library
            else:
                lib = external_dep

            if lib is not None and lib.to_link:
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
        command.extend(["-fPIC", "-shared", "-o", str(compiled_library_file)])
        command.extend([str(fortran_obj), str(capi_obj)])
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

        self.__libraries[signature] = (capi_name, fortran_obj, compiled_library_file)

    def compile_fortran(self, name, fortran_function, directory):
        """
        Compiles Fortran source files using gfortran.

        Parameters:
            name (str): Base name for the output object file.
            fortran_sources (list): List of Fortran source file paths.
        Returns:
            Path: Path to the compiled object file.
        """

        fortran_src = directory / f"{name}_src.f90"
        fortran_src.write_text(fortran_function.get_code())

        output = directory / f"{name}_fortran.o"

        libraries = []
        libraries_dirs = []
        include_dirs = []
        additional_flags = []
        for external_dep in fortran_function.get_external_dependencies().values():
            lib = None
            if hasattr(external_dep, "library"):
                if external_dep.library is not None:
                    lib = external_dep.library
            else:
                lib = external_dep

            if lib is not None:
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

        command = ["gfortran"]
        command.extend(["-fopenmp"])
        command.extend(self.compile_flags)
        command.extend(["-fPIC", "-c", "-o", str(output)])
        command.append(str(fortran_src))
        command.extend([f"-l{lib}" for lib in libraries])
        command.extend([f"-L{lib_dir}" for lib_dir in libraries_dirs])
        command.extend([f"-I{inc_dir}" for inc_dir in include_dirs])
        command.extend(additional_flags)

        sp_run = sp.run(
            command,
            cwd=directory,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
        )

        if sp_run.returncode != 0:
            error_message = "Error while compiling:\n"
            error_message += textwrap.indent(sp_run.stdout.decode("utf-8"), "    ")
            error_message += textwrap.indent(sp_run.stderr.decode("utf-8"), "    ")
            raise Warning(error_message)

        return output

    def load_function(self, signature):
        if signature in self.__loaded_functions:
            return

        if signature not in self.__libraries:
            self.compile_function(signature)

        capi_name, _, compiled_library_file = self.__libraries[signature]
        spec = importlib.util.spec_from_file_location(capi_name, compiled_library_file)
        compiled_sub = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compiled_sub)

        name = self.get_name(signature)
        self.__loaded_functions[signature] = getattr(compiled_sub, name)

    def execute_function(self, signature, runtime_args):
        if signature not in self.__loaded_functions:
            self.load_function(signature)
        return self.__loaded_functions[signature](*runtime_args)
