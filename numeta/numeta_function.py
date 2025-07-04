import numpy as np
import subprocess as sp
from pathlib import Path
import tempfile
import importlib.util
import numpy as np
import sys
import sysconfig
import pickle
import shutil

from .builder_helper import BuilderHelper
from .syntax import Subroutine, Variable
from .datatype import DataType, ArrayType, get_datatype, size_t
import textwrap
from .capi_interface import CAPIInterface
from .types_hint import comptime


class ArgumentPlaceholder:
    """
    This class is used to store the details of the arguments of the function.
    The ones that are compile-time are stored in the is_comptime attribute.
    """

    def __init__(
        self,
        name,
        is_comptime=False,
        datatype=None,
        shape=None,
        value=False,
        fortran_order=False,
        comptime_value=None,
    ) -> None:
        self.name = name
        self.is_comptime = is_comptime
        self.comptime_value = comptime_value
        self.datatype = datatype
        self.shape = shape
        self.value = value
        self.fortran_order = fortran_order

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
        self,
        func,
        directory=None,
        do_checks=True,
        symbolic_only=False,
        compile_flags="-O3 -march=native",
        namer=None,
    ) -> None:
        self.name = func.__name__
        if directory is None:
            directory = tempfile.mkdtemp()
        self.directory = Path(directory).absolute()
        self.directory.mkdir(exist_ok=True)
        self.do_checks = do_checks
        self.symbolic_only = symbolic_only
        self.compile_flags = compile_flags.split()
        self.namer = namer

        self.__func = func
        self.__symbolic_functions = {}
        self.__compiled_functions = {}
        self.__libraries = {}

        self.comptime_args_indices = self.get_comptime_args_idx(func)

    def get_comptime_args_idx(self, func):
        return [i for i, arg in enumerate(func.__annotations__.values()) if arg is comptime]

    def get_symbolic_functions(self):
        return list(self.__symbolic_functions.values())

    def dump(self, directory):
        """
        Dumps the compiled function to a file.
        """
        directory = Path(directory)
        directory.mkdir(exist_ok=True)

        # Copy the libraries to the new directory
        new_libraries = {}
        for comptime_args, (name, library_name, library_file) in self.__libraries.items():
            new_library_file = directory / library_file.parent.name / library_file.name
            new_library_file.parent.mkdir(exist_ok=True)
            shutil.copy(library_file, new_library_file)
            new_libraries[comptime_args] = (name, library_name, new_library_file)

        with open(directory / f"{self.name}.pkl", "wb") as f:
            pickle.dump(self.__symbolic_functions, f)
            pickle.dump(new_libraries, f)

    def load(self, directory):
        """
        Loads the compiled function from a file.
        """
        with open(Path(directory) / f"{self.name}.pkl", "rb") as f:
            self.__symbolic_functions = pickle.load(f)
            self.__libraries = pickle.load(f)

        for comptime_args, (name, library_name, library_file) in self.__libraries.items():
            self.__compiled_functions[comptime_args] = self.load_compiled_function(
                name, library_name, library_file
            )

    def get_runtime_args_and_spec(self, args):

        runtime_args = []
        runtime_args_spec = []
        for i, arg in enumerate(args):
            if i in self.comptime_args_indices:
                continue
            elif isinstance(arg, np.ndarray):
                runtime_args_spec.append((arg.dtype, len(arg.shape), np.isfortran(arg)))
            elif isinstance(arg, (int, float, complex)):
                runtime_args_spec.append((type(arg),))
            runtime_args.append(arg)

        return runtime_args, tuple(runtime_args_spec)

    def __call__(self, *args):

        # TODO: probably overhead, to do in C?
        comptime_args = []
        runtime_args = []
        for i, arg in enumerate(args):
            if i in self.comptime_args_indices:
                comptime_args.append(arg)
            else:
                # The other arguments are runtime arguments and we need to
                # check their type and shape to create the Fortran interface.
                if isinstance(arg, np.generic):
                    # it is a numpy scalar like np.int32(1) or np.float64(1.0)
                    comptime_args.append((arg.dtype,))
                elif isinstance(arg, np.ndarray):
                    if arg.shape == ():
                        # it is a numpy 0-dimensional array like np.array(1)
                        comptime_args.append((arg.dtype,))
                    else:
                        comptime_args.append((arg.dtype, len(arg.shape), np.isfortran(arg)))
                elif isinstance(arg, (int, float, complex)):
                    comptime_args.append((type(arg),))
                elif isinstance(arg, ArrayType):
                    comptime_args.append((arg.dtype.get_numpy(), len(arg.shape), False))
                elif isinstance(arg, type) and issubclass(arg, DataType):
                    comptime_args.append((arg.get_numpy(),))
                else:
                    raise ValueError(f"Argument {i} of type {type(arg)} is not supported")
                runtime_args.append(arg)

        comptime_args = tuple(comptime_args)

        if self.symbolic_only:
            # If symbolic_only is True, we only return the symbolic function
            symbolic_fun = self.__symbolic_functions.get(comptime_args, None)
            if symbolic_fun is None:
                comptime_args_spec = self.get_comptime_args_spec(comptime_args)
                symbolic_fun = self.construct_function(
                    f"{self.name}_{len(self.__symbolic_functions)}", comptime_args_spec
                )
                self.__symbolic_functions[comptime_args] = symbolic_fun
            return symbolic_fun

        if comptime_args not in self.__compiled_functions:
            if self.namer is None:
                name = f"{self.name}_{len(self.__compiled_functions)}"
            else:
                name = self.namer(comptime_args)
            comptime_args_spec = self.get_comptime_args_spec(comptime_args)

            symbolic_fun = self.__symbolic_functions.get(comptime_args, None)
            if symbolic_fun is None:
                symbolic_fun = self.construct_function(name, comptime_args_spec)
                self.__symbolic_functions[comptime_args] = symbolic_fun

            library_name, library_file = self.compile_function(
                name, symbolic_fun, comptime_args_spec
            )
            self.__libraries[comptime_args] = (name, library_name, library_file)
            self.__compiled_functions[comptime_args] = self.load_compiled_function(
                name, library_name, library_file
            )
        return self.execute_function(comptime_args, runtime_args)

    def get_comptime_args_spec(self, comptime_args):
        comptime_args_spec = []
        for i, arg in enumerate(comptime_args):
            if i in self.comptime_args_indices:
                ap = ArgumentPlaceholder(f"in_{i}", is_comptime=True, comptime_value=arg)
            else:
                dtype = get_datatype(arg[0])
                if len(arg) == 1:
                    # it is a numeric type or a string
                    ap = ArgumentPlaceholder(f"in_{i}", datatype=dtype, value=dtype.can_be_value())
                else:
                    fortran_order = arg[2]
                    shape = tuple([None] * arg[1])
                    ap = ArgumentPlaceholder(
                        f"in_{i}", datatype=dtype, shape=shape, fortran_order=fortran_order
                    )

            comptime_args_spec.append(ap)
        return comptime_args_spec

    def construct_function(self, name, comptime_args_spec):
        return self.get_fortran_symb_code(name, comptime_args_spec)

    def compile_function(
        self,
        name,
        symbolic_function,
        comptime_args_spec,
    ):
        """
        Compiles Fortran code and constructs a C API interface,
        then compiles them into a shared library and loads the module.

        Parameters:
            *args: Arguments to pass to compile_fortran_function.

        Returns:
            tuple: (compiled function, subroutine)
        """

        local_dir = self.directory / name
        local_dir.mkdir(exist_ok=True)

        fortran_obj = self.compile_fortran(name, symbolic_function, local_dir)

        capi_name = f"{name}_capi"
        capi_interface = CAPIInterface(
            name,
            capi_name,
            comptime_args_spec,
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
        additional_flags = []

        for external_dep in symbolic_function.get_external_dependencies().values():
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

        return capi_name, compiled_library_file

    def load_compiled_function(self, name, capi_name, compiled_library_file):
        spec = importlib.util.spec_from_file_location(capi_name, compiled_library_file)
        compiled_sub = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compiled_sub)

        return getattr(compiled_sub, name)

    def execute_function(self, comptime_args, runtime_args):
        f = self.__compiled_functions[comptime_args]
        return f(*runtime_args)

    def get_fortran_symb_code(self, name, comptime_args):
        sub = Subroutine(name)
        builder = BuilderHelper(sub, self.__func)

        symbolic_args = []
        for arg in comptime_args:
            if arg.is_comptime:
                symbolic_args.append(arg.comptime_value)
            else:
                ftype = arg.datatype.get_fortran()
                if arg.shape is None:
                    intent = "in" if arg.datatype.can_be_value() else "inout"

                    symbolic_args.append(
                        Variable(arg.name, ftype=ftype, fortran_order=False, intent=intent)
                    )

                else:
                    dim_var = builder.generate_local_variables(
                        f"fc_n",
                        ftype=size_t.get_fortran(bind_c=True),
                        intent="in",
                        dimension=len(arg.shape),
                    )
                    sub.add_variable(dim_var)

                    symbolic_args.append(
                        Variable(
                            arg.name,
                            ftype=ftype,
                            fortran_order=arg.fortran_order,
                            dimension=tuple([dim_var[i] for i in range(len(arg.shape))]),
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
