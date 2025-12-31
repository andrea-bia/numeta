import numpy as np
from pathlib import Path
import tempfile
import importlib.util
import sys
import sysconfig
import pickle
import shutil

from numeta.external_library import ExternalLibrary

from .compiler import FortranCompiler
from .registry import register_compilation_target
from .settings import settings
from .builder_helper import BuilderHelper
from .syntax import Subroutine, Variable
from .datatype import size_t
from .capi_interface import CAPIInterface
from .array_shape import ArrayShape, SCALAR, UNKNOWN


from .signature import (
    convert_signature_to_argument_specs,
    get_signature_and_runtime_args,
    parse_function_parameters,
)


class NumetaCompilationTarget(ExternalLibrary):
    """
    To link NumetaFunctions when called by other NumetaFunctions
    """

    def __init__(
        self,
        name,
        symbolic_function,
        *,
        directory: None | Path = None,
        do_checks=False,
        compile_flags="-O3 -march=native",
    ):
        super().__init__(name, to_link=False)
        self.symbolic_function = symbolic_function
        if directory is None:
            directory = tempfile.mkdtemp()
        self.directory = Path(directory).absolute()
        self.directory.mkdir(exist_ok=True)
        self.do_checks = do_checks
        if isinstance(compile_flags, str):
            self.compile_flags = compile_flags.split()
        else:
            self.compile_flags = compile_flags

        self._obj_files = None
        self.compiled_with_capi_file = None
        self.capi_name = None

        # Register this instance for later inspection via the class registry
        register_compilation_target(self)

        # Object files of all the NumetaCompilationTarget needed by this one
        # Because to link (capi part) we need all the obj files
        self._nested_obj_files = None

        # Fortunately we dont need nested mod files to compile

    @property
    def obj_files(self):
        if self._obj_files is None:
            self._obj_files, self._include = self.compile_fortran()
        return [self._obj_files]

    @property
    def include(self):
        if self._obj_files is None:
            self._obj_files, self._include = self.compile_fortran()
        return [self._include]

    def copy(self, directory):
        result = NumetaCompilationTarget(
            self.name,
            self.symbolic_function,
            directory=directory,
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

    def compile_fortran(self):
        """
        Compile Fortran source files using gfortran and return the resulting object file.
        """
        compiler = FortranCompiler(self.compile_flags)
        return compiler.compile_fortran(
            name=self.name,
            directory=self.directory,
            source=self.symbolic_function.get_code(),
            dependencies=self.symbolic_function.get_dependencies().values(),
        )

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

        self.compiled_with_capi_file = self.directory / f"lib{self.name}_module.so"

        libraries = [
            "gfortran",
            "mvec",  # link math vectorized version
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

        compiler = FortranCompiler(self.compile_flags)
        compiler.run_command(command, cwd=self.directory)

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
        self.compile_flags = compile_flags

        self.namer = namer
        self.inline = inline
        self._func = func

        # To store the dependencies of the compiled functions to other numeta generated functions.
        (
            self.params,
            self.fixed_param_indices,
            self.n_positional_or_default_args,
            self.catch_var_positional_name,
        ) = parse_function_parameters(func)

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

    def get_signature(self, *args, **kwargs):
        _, signature, _ = get_signature_and_runtime_args(
            args,
            kwargs,
            params=self.params,
            fixed_param_indices=self.fixed_param_indices,
            n_positional_or_default_args=self.n_positional_or_default_args,
            catch_var_positional_name=self.catch_var_positional_name,
        )
        return signature

    def __call__(self, *args, **kwargs):
        """
        **Note**: Support for fixed shapes for arrays works when not using numpy arrays.
        But if the function is called with numeta types hint like nm.float32[2, 3] it will create a symbolic function with fixed shapes arguments.
        Calling it is another story and not implemented yet.
        """

        builder = BuilderHelper.current_builder
        if builder is not None:
            # We are already contructing a symbolic function

            _, signature, runtime_args = get_signature_and_runtime_args(
                args,
                kwargs,
                params=self.params,
                fixed_param_indices=self.fixed_param_indices,
                n_positional_or_default_args=self.n_positional_or_default_args,
                catch_var_positional_name=self.catch_var_positional_name,
            )

            symbolic_fun = self.get_symbolic_function(signature)
            return_specs = self.return_signatures.get(signature, [])

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

            return_arguments = []
            return_values = []
            return_pointers = []
            if return_specs:
                for dtype, rank in return_specs:
                    if rank == 0:
                        out_var = builder.generate_local_variables(
                            "fc_r",
                            ftype=dtype.get_fortran(),
                        )
                        return_arguments.append(out_var)
                        return_values.append(out_var)
                        continue

                    shape_var = builder.generate_local_variables(
                        "fc_out_shape",
                        ftype=size_t.get_fortran(bind_c=True),
                        shape=ArrayShape((rank,)),
                    )
                    return_arguments.append(shape_var)

                    array_shape = ArrayShape(tuple([None] * rank))

                    if settings.use_numpy_allocator:
                        from numeta.external_modules.iso_c_binding import FPointer_c, iso_c

                        out_ptr = builder.generate_local_variables(
                            "fc_out_ptr",
                            ftype=FPointer_c,
                        )
                        return_arguments.append(out_ptr)

                        out_array = builder.generate_local_variables(
                            "fc_r",
                            ftype=dtype.get_fortran(),
                            shape=array_shape,
                            pointer=True,
                        )
                        return_pointers.append((out_ptr, out_array, shape_var, rank))
                        return_values.append(out_array)
                    else:
                        out_array = builder.generate_local_variables(
                            "fc_r",
                            ftype=dtype.get_fortran(),
                            shape=array_shape,
                            allocatable=True,
                        )
                        return_arguments.append(out_array)
                        return_values.append(out_array)

            if do_inline:
                builder.inline(symbolic_fun, *full_runtime_args, *return_arguments)
            else:
                # This add a Call statement to the current builder
                symbolic_fun(*full_runtime_args, *return_arguments)

            for out_ptr, out_array, shape_var, rank in return_pointers:
                if rank == 1:
                    shape_fortran = shape_var
                else:
                    shape_fortran = shape_var[rank - 1 : 1 : -1]
                from numeta.external_modules.iso_c_binding import iso_c

                iso_c.c_f_pointer(out_ptr, out_array, shape_fortran)

            if return_specs:
                if len(return_values) == 1:
                    return return_values[0]
                return tuple(return_values)
        else:

            # TODO: probably overhead, to do in C?
            to_execute, signature, runtime_args = get_signature_and_runtime_args(
                args,
                kwargs,
                params=self.params,
                fixed_param_indices=self.fixed_param_indices,
                n_positional_or_default_args=self.n_positional_or_default_args,
                catch_var_positional_name=self.catch_var_positional_name,
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
        argument_specs = convert_signature_to_argument_specs(
            signature,
            params=self.params,
            fixed_param_indices=self.fixed_param_indices,
            n_positional_or_default_args=self.n_positional_or_default_args,
        )

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
            directory=self.directory / self.get_name(signature),
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
            args_details=convert_signature_to_argument_specs(
                signature,
                params=self.params,
                fixed_param_indices=self.fixed_param_indices,
                n_positional_or_default_args=self.n_positional_or_default_args,
            ),
            return_specs=self.return_signatures[signature],
            directory=self.directory / self.get_name(signature),
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
