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
from .syntax.expressions import GetItem, GetAttr
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
        rank=0,
        shape=None,
        value=False,
        fortran_order=False,
        comptime_value=None,
    ) -> None:
        self.name = name
        self.is_comptime = is_comptime
        self.comptime_value = comptime_value
        self.datatype = datatype
        self.rank = rank
        self.shape = shape
        if self.shape is None:
            self.shape = tuple([None] * self.rank)
        self.value = value
        self.fortran_order = fortran_order

    @property
    def und_shape(self):
        """
        Returns the indices of the dimensions that are undefined at compile time.
        """
        if self.rank == 0:
            return []
        return [i for i, dim in enumerate(self.shape) if dim is None]

    def has_und_dims(self):
        """
        Checks if the argument has undefined dimensions at compile time.
        """
        return None in self.shape


class NumetaFunction:
    def __init__(
        self,
        func,
        directory=None,
        do_checks=True,
        compile_flags="-O3 -march=native",
        namer=None,
    ) -> None:
        self.name = func.__name__
        if directory is None:
            directory = tempfile.mkdtemp()
        self.directory = Path(directory).absolute()
        self.directory.mkdir(exist_ok=True)
        self.do_checks = do_checks
        self.compile_flags = compile_flags.split()
        self.namer = namer

        self.__func = func
        self.__comptime_args_to_name = {}
        self.__symbolic_functions = {}
        self.__libraries = {}
        self.__loaded_functions = {}

        # To store the dependencies of the compiled functions to other numeta generated functions.
        self.__dependencies = {}

        self.comptime_args_indices = self.get_comptime_args_idx(func)

    def get_name(self, comptime_args):
        if comptime_args not in self.__comptime_args_to_name:
            if self.namer is None:
                name = f"{self.name}_{len(self.__comptime_args_to_name)}"
            else:
                name = self.namer(comptime_args)
            self.__comptime_args_to_name[comptime_args] = name
        return self.__comptime_args_to_name[comptime_args]

    def run_symbolic(self, *args):
        return self.__func(*args)

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
        for comptime_args, (lib_name, lib_obj, lib_file) in self.__libraries.items():
            name = self.get_name(comptime_args)
            new_directory = directory / name
            new_directory.mkdir(exist_ok=True)
            new_lib_obj = new_directory / lib_obj.name
            new_lib_file = new_directory / lib_file.name
            shutil.copy(lib_obj, new_lib_obj)
            shutil.copy(lib_file, new_lib_file)
            new_libraries[comptime_args] = (lib_name, new_lib_obj, new_lib_file)

        with open(directory / f"{self.name}.pkl", "wb") as f:
            pickle.dump(self.__comptime_args_to_name, f)
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
            self.__comptime_args_to_name = pickle.load(f)
            self.__symbolic_functions = pickle.load(f)
            self.__libraries = pickle.load(f)

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

    def get_comptime_and_runtime_args(self, args):
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
                    comptime_args.append((arg.dtype, len(arg.shape), np.isfortran(arg)))
                elif isinstance(arg, (int, float, complex)):
                    comptime_args.append((type(arg),))
                elif isinstance(arg, ArrayType):
                    if arg.shape is None:
                        # it is a pointer
                        raise NotImplementedError("Pointers are not supported yet")
                    # add the shape only of is it fixed
                    if None in arg.shape:
                        comptime_args.append((arg.dtype.get_numpy(), len(arg.shape), False))
                        # Should add runtime args?
                        raise NotImplementedError
                    else:
                        comptime_args.append(
                            (arg.dtype.get_numpy(), len(arg.shape), False, arg.shape)
                        )
                elif isinstance(arg, type) and issubclass(arg, DataType):
                    comptime_args.append((arg.get_numpy(),))
                elif isinstance(arg, Variable):
                    ftype = arg.ftype
                    dtype = DataType.from_ftype(ftype)
                    if arg.dimension is None:
                        if arg.intent == "in":
                            # it should be value
                            comptime_args.append((dtype.get_numpy(),))
                        else:
                            comptime_args.append((dtype.get_numpy(), 0))
                    else:
                        is_fixed = True
                        shape = []
                        for dim in arg.dimension:
                            if isinstance(dim, int):
                                shape.append(dim)
                            else:
                                shape.append(None)
                                is_fixed = False
                        shape = tuple(shape)

                        if is_fixed:
                            comptime_args.append(
                                (dtype.get_numpy(), len(shape), arg.fortran_order, shape)
                            )
                        else:
                            comptime_args.append((dtype.get_numpy(), len(shape), arg.fortran_order))
                            runtime_args.append(arg.get_shape_array())

                elif isinstance(arg, GetItem):
                    ftype = arg.variable.ftype
                    dtype = DataType.from_ftype(ftype)
                    if arg.sliced is None:
                        comptime_args.append((dtype.get_numpy(), None))
                        # TODO means what, pointer?
                        raise NotImplementedError
                    elif isinstance(arg.sliced, tuple):
                        to_add_descriptor = False
                        for dim in arg.sliced:
                            if isinstance(dim, slice):
                                to_add_descriptor = True

                        if to_add_descriptor:
                            rank = len(arg.sliced)
                            comptime_args.append((dtype.get_numpy(), rank))
                            runtime_args.append(arg.get_shape_array())
                        else:
                            comptime_args.append((dtype.get_numpy(), 0))
                    else:
                        raise ValueError("sliced can be only None or slice")

                elif isinstance(arg, GetAttr):
                    struct_ftype = arg.variable.ftype
                    struct_dtype = DataType.from_ftype(struct_ftype)
                    dtype = None
                    shape = None
                    for member in struct_dtype.members:
                        if member[0] == arg.attr:
                            dtype = member[1]
                            if len(member) > 2:
                                shape = member[2]
                            break
                    if dtype is None:
                        raise ValueError(f"Attribute {arg.attr} not found in {struct_dtype}")

                    if shape is None:
                        # Put shape = 0 to make it not value
                        comptime_args.append((dtype.get_numpy(), 0))
                    else:
                        # The shape of a struct member is fixed
                        comptime_args.append(
                            (dtype.get_numpy(), len(shape), arg.variable.fortran_order, shape)
                        )
                else:
                    raise ValueError(f"Argument {i} of type {type(arg)} is not supported")
                runtime_args.append(arg)

        comptime_args = tuple(comptime_args)

        return comptime_args, runtime_args

    def __call__(self, *args):
        """
        **Note**: Support for fixed shapes for arrays works when not using numpy arrays.
        But if the function is called with numeta types hint like nm.float32[2, 3] it will create a symbolic function with fixed shapes arguments.
        Calling it is another story and not implemented yet.
        """

        if BuilderHelper.current_builder is not None:
            # We are already contructing a symbolic function

            comptime_args, runtime_args = self.get_comptime_and_runtime_args(args)

            symbolic_fun = self.get_symbolic_function(comptime_args)

            # This add a Call statement to the current builder
            symbolic_fun(*runtime_args)

            # Add this function to the dependencies of the current builder
            caller = BuilderHelper.get_current_builder().numeta_function
            caller_identifier = BuilderHelper.get_current_builder().comptime_args
            if caller_identifier not in caller.__dependencies:
                caller.__dependencies[caller_identifier] = {}
            caller.__dependencies[caller_identifier][self.name, comptime_args] = (
                self,
                comptime_args,
            )

        else:
            to_compile = True
            # TODO: probably overhead, to do in C?
            # a comptme arg is a tuple like this:
            # (dtype, rank, has_fortran_order, shape)
            # The last elements can be omitted
            # To be fast constructed and hashable to speed calls
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
                        comptime_args.append((arg.dtype, len(arg.shape), np.isfortran(arg)))
                    elif isinstance(arg, (int, float, complex)):
                        comptime_args.append((type(arg),))
                    elif isinstance(arg, ArrayType):
                        if arg.shape is None:
                            # it is a pointer
                            raise NotImplementedError("Pointers are not supported yet")
                        # Here we keep all the information about the array shape because we can construct a symbolic function with fixed shapes.
                        comptime_args.append(
                            (arg.dtype.get_numpy(), len(arg.shape), False, arg.shape)
                        )
                        to_compile = False
                    elif isinstance(arg, type) and issubclass(arg, DataType):
                        comptime_args.append((arg.get_numpy(),))
                        to_compile = False
                    else:
                        raise ValueError(f"Argument {i} of type {type(arg)} is not supported")
                    runtime_args.append(arg)

            comptime_args = tuple(comptime_args)

            if not to_compile:
                return self.get_symbolic_function(comptime_args)

            self.get_symbolic_function(comptime_args)

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
                    # for numpy arrays arg[1] is the rank, for the other types it is the shape
                    rank = arg[1]

                    fortran_order = False
                    if len(arg) == 3:
                        fortran_order = arg[2]

                    shape = None
                    if len(arg) == 4:
                        shape = arg[3]

                    ap = ArgumentPlaceholder(
                        f"in_{i}",
                        datatype=dtype,
                        rank=rank,
                        shape=shape,
                        fortran_order=fortran_order,
                    )

            comptime_args_spec.append(ap)
        return comptime_args_spec

    def get_symbolic_function(self, comptime_args):
        if comptime_args not in self.__symbolic_functions:
            self.construct_symbolic_function(comptime_args)
        return self.__symbolic_functions[comptime_args]

    def construct_symbolic_function(self, comptime_args):
        name = self.get_name(comptime_args)
        comptime_args_spec = self.get_comptime_args_spec(comptime_args)

        sub = Subroutine(name)
        builder = BuilderHelper(self, sub, comptime_args)

        symbolic_args = []
        for arg in comptime_args_spec:
            if arg.is_comptime:
                symbolic_args.append(arg.comptime_value)
            else:
                ftype = arg.datatype.get_fortran()
                if arg.rank == 0:
                    # is it a scalar
                    intent = "in" if arg.value else "inout"

                    symbolic_args.append(
                        Variable(arg.name, ftype=ftype, fortran_order=False, intent=intent)
                    )
                elif arg.has_und_dims():

                    # The dimension has to be passed
                    dim_var = builder.generate_local_variables(
                        f"fc_n",
                        ftype=size_t.get_fortran(bind_c=True),
                        intent="in",
                        dimension=arg.rank,
                    )
                    sub.add_variable(dim_var)

                    symbolic_args.append(
                        Variable(
                            arg.name,
                            ftype=ftype,
                            fortran_order=arg.fortran_order,
                            dimension=tuple([dim_var[i] for i in range(arg.rank)]),
                            intent="inout",
                        )
                    )
                else:
                    # The dimension is fixed
                    symbolic_args.append(
                        Variable(
                            arg.name,
                            ftype=ftype,
                            fortran_order=arg.fortran_order,
                            dimension=arg.shape,
                            intent="inout",
                        )
                    )

                sub.add_variable(symbolic_args[-1])

        builder.build(*symbolic_args)
        self.__symbolic_functions[comptime_args] = sub

    def compile_function(
        self,
        comptime_args,
    ):
        """
        Compiles Fortran code and constructs a C API interface,
        then compiles them into a shared library and loads the module.

        Parameters:
            *args: Arguments to pass to compile_fortran_function.

        Returns:
            tuple: (compiled function, subroutine)
        """
        name = self.get_name(comptime_args)

        local_dir = self.directory / name
        local_dir.mkdir(exist_ok=True)

        symbolic_function = self.get_symbolic_function(comptime_args)
        fortran_obj = self.compile_fortran(name, symbolic_function, local_dir)

        capi_name = f"{name}_capi"
        comptime_args_spec = self.get_comptime_args_spec(comptime_args)
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
        extra_objects = []
        additional_flags = []

        if comptime_args in self.__dependencies:
            for dep, dep_comptime_args in self.__dependencies[comptime_args].values():
                if dep_comptime_args not in dep.__libraries:
                    dep.compile_function(dep_comptime_args)
                _, dep_obj, _ = dep.__libraries[dep_comptime_args]
                extra_objects.append(dep_obj)

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

        self.__libraries[comptime_args] = (capi_name, fortran_obj, compiled_library_file)

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

    def load_function(self, comptime_args):
        if comptime_args in self.__loaded_functions:
            return

        if comptime_args not in self.__libraries:
            self.compile_function(comptime_args)

        capi_name, _, compiled_library_file = self.__libraries[comptime_args]
        spec = importlib.util.spec_from_file_location(capi_name, compiled_library_file)
        compiled_sub = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compiled_sub)

        name = self.get_name(comptime_args)
        self.__loaded_functions[comptime_args] = getattr(compiled_sub, name)

    def execute_function(self, comptime_args, runtime_args):
        if comptime_args not in self.__loaded_functions:
            self.load_function(comptime_args)
        f = self.__loaded_functions[comptime_args]
        return f(*runtime_args)
