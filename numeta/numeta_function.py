import inspect
import numpy as np
import subprocess as sp
from pathlib import Path
import tempfile
import importlib.util
import numpy as np
import sys
import sysconfig
from .builder_helper import BuilderHelper
from .syntax import Subroutine, Variable
from .datatype import DataType, size_t_dtype
import textwrap
from .settings import settings
from .capi_interface import CAPIInterface


class ArgumentPlaceholder:
    """
    This class is used to store the details of the arguments of the function.
    The ones that are compile-time are stored in the is_comptime attribute.
    """

    def __init__(
        self, name, is_comptime=False, datatype=None, shape=None, value=False, order="C"
    ) -> None:
        self.name = name
        self.is_comptime = is_comptime
        self.datatype = datatype
        self.shape = shape
        self.value = value
        self.order = order

    @property
    def und_shape(self):
        """
        Returns the indices of the dimensions that are undefined at compile time.
        """
        if self.shape is None or isinstance(self.shape, int):
            return []
        return [i for i, dim in enumerate(self.shape) if dim is None]

    def has_und_dims(self):
        """
        Checks if the argument has undefined dimensions at compile time.
        """
        if isinstance(self.shape, (tuple, list)):
            return None in self.shape
        return False


class NumetaFunction:
    def __init__(
        self, func, directory=None, do_checks=True, compile_flags="-O3 -march=native"
    ) -> None:
        self.name = func.__name__
        if directory is None:
            directory = tempfile.mkdtemp()
        self.directory = Path(directory).absolute()
        self.directory.mkdir(exist_ok=True)
        self.do_checks = do_checks
        self.compile_flags = compile_flags.split()

        self.__func = func
        self.__fortran_functions = {}
        self.__symbolic_functions = {}  # the symbolic representation of the function
        self.args_details = self.get_args_details(func)
        self.comptime_vars_indices = [
            i for i in range(len(self.args_details)) if self.args_details[i].is_comptime
        ]
        self.runtime_vars_indices_and_undeclared_shapes = [
            (i, self.args_details[i].und_shape)
            for i in range(len(self.args_details))
            if not self.args_details[i].is_comptime
        ]
        self.runtime_vars_indices = [
            i for i in range(len(self.args_details)) if not self.args_details[i].is_comptime
        ]

    def code(self, *args):
        if len(self.comptime_vars_indices) == 0:
            if None not in self.__fortran_functions:
                (
                    self.__fortran_functions[None],
                    self.__symbolic_functions[None],
                ) = self.compile_function(*args)
            return self.__fortran_functions[None](*args).get_code()
        else:
            comptime_args = tuple(args[i] for i in self.comptime_vars_indices)

            symbolic_fun = self.__symbolic_functions.get(comptime_args, None)
            if symbolic_fun is None:
                fun, symbolic_fun = self.compile_function(*args)
                self.__fortran_functions[comptime_args] = fun
                self.__symbolic_functions[comptime_args] = symbolic_fun
            return symbolic_fun.get_code()

    def __call__(self, *args):
        if len(self.comptime_vars_indices) == 0:
            if None not in self.__fortran_functions:
                (
                    self.__fortran_functions[None],
                    self.__symbolic_functions[None],
                ) = self.compile_function(*args)
            return self.__fortran_functions[None](*args)
        else:
            return self.call_with_comptime(*args)

    def call_with_comptime(self, *args):
        comptime_args = tuple(args[i] for i in self.comptime_vars_indices)
        runtime_args = [args[i] for i in self.runtime_vars_indices]

        fun = self.__fortran_functions.get(comptime_args, None)
        if fun is None:
            fun, symbolic_fun = self.compile_function(*args)
            self.__fortran_functions[comptime_args] = fun
            self.__symbolic_functions[comptime_args] = symbolic_fun
        return fun(*runtime_args)

    def get_args_details(self, func):
        args_details = []

        runtime_args = {}
        for name, hint in func.__annotations__.items():
            if hasattr(hint, "dtype") and isinstance(hint.dtype, DataType):
                datatype = hint.dtype
                shape = hint.flags["shape"]
                order = hint.flags.get("order", settings.order)
                if datatype.can_be_value() and shape is None:
                    runtime_args[name] = ArgumentPlaceholder(
                        name,
                        is_comptime=False,
                        datatype=datatype,
                        value=True,
                        order=order,
                    )
                elif shape is None:
                    runtime_args[name] = ArgumentPlaceholder(
                        name, is_comptime=False, datatype=datatype, order=order
                    )
                elif isinstance(shape, int):
                    runtime_args[name] = ArgumentPlaceholder(
                        name,
                        is_comptime=False,
                        datatype=datatype,
                        shape=shape,
                        order=order,
                    )
                elif isinstance(shape, slice):
                    if shape.start is None and shape.stop is None and shape.step is None:
                        runtime_args[name] = ArgumentPlaceholder(
                            name,
                            is_comptime=False,
                            datatype=datatype,
                            shape=[None],
                            order=order,
                        )
                    else:
                        raise ValueError('Only ":" is allowed for slice')
                elif isinstance(shape, tuple):
                    parsed_shape = []
                    for dim in shape:
                        if isinstance(dim, slice):
                            if dim.start is None and dim.stop is None and dim.step is None:
                                parsed_shape.append(None)
                            else:
                                raise ValueError('Only ":" is allowed for slice')
                        else:
                            parsed_shape.append(dim)
                    runtime_args[name] = ArgumentPlaceholder(
                        name,
                        is_comptime=False,
                        datatype=datatype,
                        shape=parsed_shape,
                        order=order,
                    )

        params = inspect.signature(func).parameters

        args_details = [
            runtime_args.get(key, ArgumentPlaceholder(key, is_comptime=True)) for key in params
        ]

        return args_details

    def compile_function(
        self,
        *args,
    ):
        """
        Compiles Fortran code and constructs a C API interface,
        then compiles them into a shared library and loads the module.

        Parameters:
            *args: Arguments to pass to compile_fortran_function.

        Returns:
            tuple: (compiled function, subroutine)
        """

        local_dir = self.directory / f"{len(self.__fortran_functions)}"
        local_dir.mkdir(exist_ok=True)

        fortran_function = self.get_fortran_symb_code(*args)
        fortran_obj = self.compile_fortran(self.name, fortran_function, local_dir)

        capi_name = f"{self.name}_capi_{len(self.__fortran_functions)}"
        capi_interface = CAPIInterface(
            self.name,
            capi_name,
            self.args_details,
            local_dir,
            self.compile_flags,
            self.do_checks,
        )
        capi_obj = capi_interface.generate()

        compiled_library_file = local_dir / f"lib{self.name}_module.so"

        libraries = [
            "gfortran",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
        ]
        libraries_dirs = []
        include_dirs = [sysconfig.get_paths()["include"], np.get_include()]
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

        command = ["gcc"]
        command.extend(self.compile_flags)
        command.extend(["-fopenmp"])
        command.extend(["-fPIC", "-shared", "-o", str(compiled_library_file)])
        command.extend([str(fortran_obj), str(capi_obj)])
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

        spec = importlib.util.spec_from_file_location(capi_name, compiled_library_file)
        compiled_sub = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compiled_sub)

        return getattr(compiled_sub, self.name), fortran_function

    def get_fortran_symb_code(self, *args):
        sub = Subroutine(self.name)
        builder = BuilderHelper(sub, self.__func)

        symbolic_args = []
        for i, arg in enumerate(self.args_details):
            if arg.is_comptime:
                symbolic_args.append(args[i])
            else:
                ftype = arg.datatype.get_fortran()
                is_fortran = arg.order == "F"
                if arg.value:
                    symbolic_args.append(
                        Variable(arg.name, ftype=ftype, fortran_order=is_fortran, intent="in")
                    )
                elif arg.shape is None:
                    symbolic_args.append(
                        Variable(
                            arg.name,
                            ftype=ftype,
                            fortran_order=is_fortran,
                            intent="inout",
                        )
                    )
                elif isinstance(arg.shape, int):
                    symbolic_args.append(
                        Variable(
                            arg.name,
                            ftype=ftype,
                            fortran_order=is_fortran,
                            intent="inout",
                            dimension=arg.shape,
                        )
                    )
                else:
                    dim_var = builder.generate_local_variables(
                        f"fc_n",
                        ftype=size_t_dtype.get_fortran(bind_c=True),
                        intent="in",
                        dimension=len(arg.shape),
                    )
                    to_add = False
                    dimension = []
                    for i_dim, dim in enumerate(arg.shape):
                        if dim is None:
                            dimension.append(dim_var[i_dim])
                            to_add = True
                        else:
                            dimension.append(dim)

                    if to_add:
                        sub.add_variable(dim_var)

                    symbolic_args.append(
                        Variable(
                            arg.name,
                            ftype=ftype,
                            fortran_order=is_fortran,
                            dimension=tuple(dimension),
                            intent="inout",
                        )
                    )
                sub.add_variable(symbolic_args[-1])

        builder.build(*symbolic_args)

        return sub

    def compile_fortran(self, name, fortran_function, directory):
        """
        Compiles Fortran source files using gfortran.

        Parameters:
            name (str): Base name for the output object file.
            fortran_sources (list): List of Fortran source file paths.
        Returns:
            Path: Path to the compiled object file.
        """

        fortran_src = directory / f"{self.name}_src.f90"
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
