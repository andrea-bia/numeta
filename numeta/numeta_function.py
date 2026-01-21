import numpy as np
from pathlib import Path
import tempfile
import importlib.util
import sys
import sysconfig
import pickle
import shutil

from .compiler import Compiler
from .settings import settings
from .builder_helper import BuilderHelper
from .syntax import Subroutine, Variable
from .datatype import size_t
from .capi_interface import CAPIInterface
from .array_shape import ArrayShape, SCALAR, UNKNOWN
from .external_library import ExternalLibrary
from .signature import (
    convert_signature_to_argument_specs,
    get_signature_and_runtime_args,
    parse_function_parameters,
)


class NumetaCompiledFunction(ExternalLibrary):

    def __init__(
        self,
        name,
        symbolic_function,
        *,
        path: None | Path = None,
        do_checks=False,
        compile_flags="-O3 -march=native",
    ):
        """
        Has to be linked at runtime
        """
        super().__init__(name, to_link=True)
        self.name2 = name
        self.symbolic_function = symbolic_function
        if path is None:
            path = tempfile.mkdtemp()
        self._path = Path(path).absolute()
        self._path.mkdir(exist_ok=True)
        self._rpath = self._path
        self.do_checks = do_checks
        if isinstance(compile_flags, str):
            self.compile_flags = compile_flags.split()
        else:
            self.compile_flags = compile_flags
        self.compiled = False

    @property
    def obj_files(self):
        if self._obj_files is None:
            self._obj_files, self._include = self.compile_obj()
        return [str(self._obj_files)]

    @property
    def include(self):
        if self._obj_files is None:
            self._obj_files, self._include = self.compile_obj()
        return [self._include]

    @property
    def path(self):
        if not self.compiled:
            self.compile()
        return str(self._path)

    @property
    def rpath(self):
        if not self.compiled:
            self.compile()
        return str(self._rpath)

    def compile_obj(self):
        """
        Compile Fortran source files using gfortran and return the resulting object file.
        """
        if self._obj_files is None:
            compiler = Compiler("gfortran", self.compile_flags)
            fortran_src = self._path / f"{self.name}_src.f90"
            fortran_src.write_text(self.symbolic_function.get_code())

            include_dirs = set()
            additional_flags = set()

            for lib in self.symbolic_function.get_dependencies().values():

                if lib.include is not None:
                    if isinstance(lib.include, (list, tuple, set)):
                        include_dirs |= set(lib.include)
                    else:
                        include_dirs.add(lib.include)

                if lib.additional_flags is not None:
                    if isinstance(lib.additional_flags, str):
                        additional_flags.add(tuple(lib.additional_flags.split()))
                    else:
                        additional_flags.add(tuple(lib.additional_flags))

            return compiler.compile_to_obj(
                name=self.name,
                directory=self._path,
                sources=[str(fortran_src)],
                include_dirs=include_dirs,
                additional_flags=additional_flags,
            )

    def compile(self):
        """
        Compile core lib (no wrapper)
        """
        if not self.compiled:

            # find dependencies

            libraries = {"gfortran", "mvec"}
            libraries_dirs = set()
            rpath_dirs = set()
            include_dirs = set()
            additional_flags = set()

            for lib in self.symbolic_function.get_dependencies().values():

                if lib.include is not None:
                    if isinstance(lib.include, (list, tuple, set)):
                        include_dirs |= set(lib.include)
                    else:
                        include_dirs.add(lib.include)

                if lib.to_link:
                    libraries.add(lib.name)
                    if lib.path is not None:
                        libraries_dirs.add(str(lib.path))
                    if lib.rpath is not None:
                        rpath_dirs.add(str(lib.rpath))

                if lib.additional_flags is not None:
                    if isinstance(lib.additional_flags, str):
                        additional_flags.add(tuple(lib.additional_flags.split()))
                    else:
                        additional_flags.add(tuple(lib.additional_flags))

            compiler = Compiler("gcc", self.compile_flags)
            lib = compiler.compile_to_library(
                self.name,
                self.obj_files,
                self._path,
                libraries=libraries,
                libraries_dirs=libraries_dirs,
                rpath_dirs=rpath_dirs,
                include_dirs=include_dirs,
                additional_flags=additional_flags,
            )
            self.compiled = True
            return lib


class NumetaFunction:
    """
    Representation of a JIT-compiled function.
    """

    n_libs = 0

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
        self.return_signatures = {}  # Only needed if i create symbolic and after compile
        self._compiled_functions = {}
        self._capi_interfaces = {}
        self._fast_call = {}

    def get_symbolic_functions(self):
        return [v.symbolic_function for v in self._compiled_functions.values()]

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

            if signature not in self._compiled_functions:
                self.construct_compiled_target(signature)
            symbolic_fun = self._compiled_functions[signature].symbolic_function
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
                self.construct_compiled_target(signature)
                return self._compiled_functions[signature].symbolic_function

            return self.execute(signature, runtime_args)

    def get_symbolic_function(self, name, signature):
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
        return sub

    def construct_compiled_target(self, signature):

        if self.namer is None:
            name = f"{self.name}_{NumetaFunction.n_libs}"
        else:
            name = self.namer(*signature)
        NumetaFunction.n_libs += 1

        symbolic_fun = self.get_symbolic_function(name, signature)

        self._compiled_functions[signature] = NumetaCompiledFunction(
            name,
            symbolic_fun,
            path=self.directory,
            do_checks=self.do_checks,
            compile_flags=self.compile_flags,
        )

        symbolic_fun.parent = self._compiled_functions[signature]

    def construct_wrapper(self, signature):
        name = self._compiled_functions[signature].name

        procedures_infos = [
            (
                name,
                convert_signature_to_argument_specs(
                    signature,
                    params=self.params,
                    fixed_param_indices=self.fixed_param_indices,
                    n_positional_or_default_args=self.n_positional_or_default_args,
                ),
                self.return_signatures[signature],
            )
        ]
        self._capi_interfaces[signature] = CAPIInterface(
            name=name,
            functions=procedures_infos,
        )

        return self._capi_interfaces[signature]

    def compile(self, signature):
        if not self._compiled_functions[signature].compiled:
            self._compiled_functions[signature].compile()

        if self._capi_interfaces[signature].lib_path is None:
            wrapper_src = self.directory / f"{self._capi_interfaces[signature].name}.c"
            self._capi_interfaces[signature].write(wrapper_src)

            libraries = [
                "gfortran",
                "mvec",  # link math vectorized version
                f"python{sys.version_info.major}.{sys.version_info.minor}",
                self._compiled_functions[signature].name,
            ]
            lib_dirs = []
            rpath_dirs = [self._compiled_functions[signature].path]
            include_dirs = [sysconfig.get_paths()["include"], np.get_include()]
            additional_flags = ["-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"]

            compiler = Compiler("gcc", self.compile_flags)
            lib = compiler.compile_to_library(
                self._capi_interfaces[signature].name,
                [wrapper_src],
                directory=self.directory,
                include_dirs=include_dirs,
                libraries=libraries,
                libraries_dirs=[],
                rpath_dirs=rpath_dirs,
                additional_flags=additional_flags,
            )
            self._capi_interfaces[signature].set_lib_path(lib)

    def load(self, signature):
        if signature not in self._compiled_functions:
            self.construct_compiled_target(signature)
        if signature not in self._capi_interfaces:
            self.construct_wrapper(signature)
        if self._capi_interfaces[signature].lib_path is None:
            self.compile(signature)
        spec = importlib.util.spec_from_file_location(
            self._capi_interfaces[signature].name, self._capi_interfaces[signature].lib_path
        )
        compiled_sub = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compiled_sub)
        self._fast_call[signature] = getattr(
            compiled_sub, self._compiled_functions[signature].name2
        )

    def execute(self, signature, runtime_args):
        if signature not in self._fast_call:
            self.load(signature)
        return self._fast_call[signature](*runtime_args)
