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
from .datatype import DataType, size_t_dtype
import textwrap
from .capi_interface import CAPIInterface
from .types_hint import comptime


class ArgumentPlaceholder:
    """
    This class is used to store the details of the arguments of the function.
    The ones that are compile-time are stored in the is_comptime attribute.
    """

    def __init__(
        self, name, is_comptime=False, datatype=None, shape=None, value=False, fortran_order=False, comptime_value=None
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
        self.__symbolic_functions = {}  # the symbolic representation of the function
        self.__fortran_functions = {}
        self.__libraries = {}

        self.comptime_args_indices = self.get_comptime_args_idx(func)

    def get_comptime_args_idx(self, func):
        return [i for i, arg in enumerate(func.__annotations__.values()) if arg is comptime]

    def dump(self, directory):
        """
        Dumps the compiled function to a file.
        """
        directory = Path(directory)
        directory.mkdir(exist_ok=True)

        # Copy the libraries to the new directory
        new_libraries = {}
        for comptime_args, (library_name, library_file) in self.__libraries.items():
            new_library_file = directory / library_file.parent.name / library_file.name
            new_library_file.parent.mkdir(exist_ok=True)
            shutil.copy(library_file, new_library_file)
            new_libraries[comptime_args] = (library_name, new_library_file)

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

        for comptime_args, (library_name, library_file) in self.__libraries.items():
            self.__fortran_functions[comptime_args] = self.load_compiled_function(
                library_name, library_file
            )

   #def code(self, *args):
   #    if len(self.comptime_vars_indices) == 0:
   #        if None not in self.__fortran_functions:
   #            library_name, library_file, symbolic_fun = self.compile_function(*args, runtime_args_spec)
   #            self.__symbolic_functions[None] = symbolic_fun
   #            self.__libraries[None] = (library_name, library_file)
   #            self.__fortran_functions[None] = self.load_compiled_function(
   #                library_name, library_file
   #            )
   #        return self.__symbolic_functions[None].get_code()
   #    else:
   #        comptime_args = tuple(args[i] for i in self.comptime_vars_indices)

   #        symbolic_fun = self.__symbolic_functions.get(comptime_args, None)
   #        if symbolic_fun is None:
   #            library_name, library_file, symbolic_fun = self.compile_function(*args)
   #            self.__symbolic_functions[comptime_args] = symbolic_fun
   #            self.__libraries[comptime_args] = (library_name, library_file)
   #            self.__fortran_functions[comptime_args] = self.load_compiled_function(
   #                library_name, library_file
   #            )
   #        return symbolic_fun.get_code()

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
                else:
                    raise ValueError(f"Argument {i} of type {type(arg)} is not supported")
                runtime_args.append(arg)

        comptime_args = tuple(comptime_args)
        
        fun = self.__fortran_functions.get(comptime_args, None)
        if fun is None:
            library_name, library_file, symbolic_fun = self.compile_function(comptime_args)
            self.__symbolic_functions[comptime_args] = symbolic_fun
            self.__libraries[comptime_args] = (library_name, library_file)
            self.__fortran_functions[comptime_args] = self.load_compiled_function(
                library_name, library_file
            )
            fun = self.__fortran_functions[comptime_args]
        return fun(*runtime_args)


    def compile_function(
        self,
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

        local_dir = self.directory / f"{len(self.__fortran_functions)}"
        local_dir.mkdir(exist_ok=True)

        comptime_args = []
        for i, arg in enumerate(comptime_args_spec):
            if i in self.comptime_args_indices:
                ap = ArgumentPlaceholder(f"in_{i}", is_comptime=True, comptime_value=arg)
            else: 
                from .types_hint import get_datatype
                dtype = get_datatype(arg[0]) 
                if len(arg) == 1:
                    # it is a numberic type or a string
                    ap = ArgumentPlaceholder(f"in_{i}", datatype=dtype, value=dtype.can_be_value())
                else:
                    fortran_order = arg[2]
                    shape = tuple([None] * arg[1])
                    ap = ArgumentPlaceholder(f"in_{i}", datatype=dtype, shape=shape, fortran_order=fortran_order)

            comptime_args.append(ap)

        fortran_function = self.get_fortran_symb_code(comptime_args)
        fortran_obj = self.compile_fortran(self.name, fortran_function, local_dir)

        capi_name = f"{self.name}_capi_{len(self.__fortran_functions)}"
        capi_interface = CAPIInterface(
            self.name,
            capi_name,
            comptime_args,
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

        return capi_name, compiled_library_file, fortran_function

    def load_compiled_function(self, capi_name, compiled_library_file):
        spec = importlib.util.spec_from_file_location(capi_name, compiled_library_file)
        compiled_sub = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compiled_sub)

        return getattr(compiled_sub, self.name)

    def get_fortran_symb_code(self, comptime_args):
        sub = Subroutine(self.name)
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
                        ftype=size_t_dtype.get_fortran(bind_c=True),
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
