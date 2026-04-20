import numpy as np
import os
import sys
import tempfile

from pathlib import Path
import pickle
import warnings
from types import MappingProxyType
from typing import Iterable

from .numeta_function import NumetaFunction, NumetaCompiledFunction
from .pyc_extension import PyCExtension
from .compiler import Compiler
from .settings import settings


class NumetaLibrary:
    __slots__ = "name", "_entries", "_global_entries"
    loaded = set()
    _reserved_names = {
        "name",
        "functions",
        "loaded",
        "register",
        "remove",
        "list_functions",
        "save",
        "load",
        "print_f90_files",
        "__getitem__",
        "__contains__",
        "__iter__",
        "__len__",
        "__repr__",
    }

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            self._nm_validate_name(name)
        self.name = name
        self._entries: dict = {}
        self._global_entries: dict = {}

    @classmethod
    def _nm_validate_name(cls, name: str) -> None:
        if name == PyCExtension.SUFFIX or name.endswith(PyCExtension.SUFFIX):
            raise ValueError(
                f"Library name '{name}' is reserved because it ends with {PyCExtension.SUFFIX}."
            )
        if name in cls.loaded:
            raise ValueError(f"Already using a library called {name}")
        if name in NumetaFunction.used_compiled_names:
            raise ValueError(
                f"Library name '{name}' conflicts with a compiled function library name."
            )
        wrapper_module = f"{name}{PyCExtension.SUFFIX}"
        if wrapper_module in sys.modules:
            raise ValueError(
                f"Library name '{name}' conflicts with an existing Numeta wrapper module."
            )

    def register(self, function: NumetaFunction) -> NumetaFunction:
        if not isinstance(function, NumetaFunction):
            raise TypeError("register expects a NumetaFunction")
        if function.name in self._reserved_names:
            raise ValueError(
                f"Function name '{function.name}' is reserved by NumetaLibrary methods"
            )
        existing = self._entries.get(function.name)
        if existing is not None and existing is not function:
            raise ValueError(f"Function '{function.name}' already registered")
        self._entries[function.name] = function
        return function

    def remove(self, name: str) -> NumetaFunction:
        return self._entries.pop(name)

    def list_functions(self) -> list[str]:
        return list(self._entries)

    @property
    def functions(self) -> MappingProxyType:
        return MappingProxyType(self._entries)

    def __getitem__(self, name: str) -> NumetaFunction:
        return self._entries[name]

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __iter__(self):
        return iter(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, size={len(self)})"

    def __getattr__(self, name) -> NumetaFunction:
        if name in self._entries:
            return self._entries[name]
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")

    def _nm_add(self, function: NumetaFunction) -> None:
        self.register(function)

    def _nm_add_global(self, global_target: NumetaCompiledFunction) -> None:
        if not isinstance(global_target, NumetaCompiledFunction):
            raise TypeError("global registration expects a NumetaCompiledFunction")
        existing = self._global_entries.get(global_target.name)
        if existing is not None and existing is not global_target:
            raise ValueError(f"Global namespace '{global_target.name}' already registered")
        self._global_entries[global_target.name] = global_target

    def _nm_get(self, name) -> NumetaFunction | None:
        return self._entries.get(name)

    def write_code(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for nm_function in self._entries.values():
            for compiled_target in nm_function._compiled_functions.values():
                if compiled_target.backend == "fortran":
                    fortran_src = directory / f"{compiled_target.name}_src.f90"
                    from .ir import FortranEmitter, lower_procedure

                    ir_proc = lower_procedure(compiled_target.symbolic_function)
                    emitter = FortranEmitter()
                    fortran_src.write_text(emitter.emit_procedure(ir_proc))
                elif compiled_target.backend == "c":
                    from numeta.c.emitter import CEmitter
                    from .ir import lower_procedure

                    c_src = directory / f"{compiled_target.name}_src.c"
                    ir_proc = lower_procedure(compiled_target.symbolic_function)
                    emitter = CEmitter()
                    c_code, _requires_math = emitter.emit_procedure(ir_proc)
                    c_src.write_text(c_code)
                else:
                    raise ValueError(f"Unsupported backend: {compiled_target.backend}")

        for global_target in self._global_entries.values():
            if global_target.backend == "fortran":
                fortran_src = directory / f"{global_target.name}_src.f90"
                from .ast.namespace import Namespace
                from .fortran.fortran_syntax import render_stmt_lines

                if not isinstance(global_target.symbolic_function, Namespace):
                    raise ValueError("Global target must be backed by a namespace")
                lines = render_stmt_lines(
                    global_target.symbolic_function.get_declaration(), indent=0
                )
                fortran_src.write_text("".join(lines))
            elif global_target.backend == "c":
                from .ast.namespace import Namespace
                from numeta.c.emitter import CEmitter

                c_src = directory / f"{global_target.name}_src.c"
                if not isinstance(global_target.symbolic_function, Namespace):
                    raise ValueError("Global target must be backed by a namespace")
                emitter = CEmitter()
                c_code, _requires_math = emitter.emit_namespace(global_target.symbolic_function)
                c_src.write_text(c_code)
            else:
                raise ValueError(f"Unsupported backend: {global_target.backend}")

    def save(
        self,
        directory: str | Path,
        compile_flags: str | Iterable[str] | None = None,
    ) -> Path:
        directory = Path(directory).absolute()
        directory.mkdir(parents=True, exist_ok=True)

        if self.name is None:
            raise ValueError("Library name must be set before saving")
        name = self.name

        #
        # Create the interface of only the functions owned by the library
        #

        procedures_infos = []
        for function in self._entries.values():
            for wrapper in function._pyc_extensions.values():
                procedures_infos.extend(wrapper.functions)

        pyc_extension = PyCExtension(
            name=self.name,
            functions=procedures_infos,
        )

        resolved_flags = settings.default_compile_flags if compile_flags is None else compile_flags
        compiler = Compiler("gcc", compile_flags=resolved_flags)

        obj_files: set[Path] = set()
        dependencies = {}
        pickle_path = directory / f"{self.name}.pkl"
        temp_pickle_path: Path | None = None

        def build_function_state(obj: NumetaFunction) -> dict:
            return {
                "name": obj.name,
                "hidden": obj.hidden,
                "external": obj.external,
                "_path": obj._path,
                "_rpath": obj._rpath,
                "_include": obj._include,
                "_obj_files": obj._obj_files,
                "additional_flags": obj.additional_flags,
                "to_link": obj.to_link,
                "namespaces": obj.namespaces,
                "procedures": obj.procedures,
                "variables": obj.variables,
                "directory": obj.directory,
                "do_checks": obj.do_checks,
                "compile_flags": obj.compile_flags,
                "backend": obj.backend,
                "namer": obj.namer,
                "inline": obj.inline,
                "_func": None,
                "params": obj.params,
                "fixed_param_indices": obj.fixed_param_indices,
                "n_positional_or_default_args": obj.n_positional_or_default_args,
                "catch_var_positional_name": obj.catch_var_positional_name,
                "return_signatures": obj.return_signatures,
                "_compiled_functions": obj._compiled_functions,
                "_pyc_extensions": obj._pyc_extensions,
                "_fast_call": {},
                "_use_c_dispatch_instance": obj._use_c_dispatch_instance,
            }

        def build_compiled_function_state(obj: NumetaCompiledFunction) -> dict:
            return {
                "name": name,
                "hidden": obj.hidden,
                "external": obj.external,
                "_path": directory,
                "_rpath": directory,
                "_include": directory,
                "_obj_files": None,
                "additional_flags": obj.additional_flags,
                "to_link": obj.to_link,
                "namespaces": obj.namespaces,
                "procedures": obj.procedures,
                "variables": obj.variables,
                "symbolic_function": obj.symbolic_function,
                "func_name": obj.func_name,
                "do_checks": obj.do_checks,
                "compile_flags": obj.compile_flags,
                "backend": obj.backend,
                "_requires_math": obj._requires_math,
                "compiled": True,
            }

        # We need to compiled ALL the NumetaFunctions not only the one directly owned by the library

        class RewritingPickler(pickle.Pickler):

            def reducer_override(self, obj):  # type: ignore[override]
                nonlocal dependencies
                nonlocal obj_files
                if isinstance(obj, NumetaFunction):
                    state = build_function_state(obj)
                    state["_pyc_extensions"] = {sig: pyc_extension for sig in obj._pyc_extensions}
                    return (NumetaFunction.__new__, (NumetaFunction,), state)

                if isinstance(obj, NumetaCompiledFunction):
                    state = build_compiled_function_state(obj)
                    obj_files.update([obj for obj in obj.obj_files])
                    dependencies |= obj.symbolic_function.get_dependencies()
                    return (NumetaCompiledFunction.__new__, (NumetaCompiledFunction,), state)
                return NotImplemented

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=directory,
                prefix=f".{self.name}.",
                suffix=".pkl.tmp",
                delete=False,
            ) as f:
                temp_pickle_path = Path(f.name)
                RewritingPickler(f).dump(list(self._entries.values()))

            libraries = set()
            libraries_dirs = set()
            rpath_dirs = set()
            include_dirs = set()
            additional_flags = set()

            for lib in dependencies.values():

                if isinstance(lib, NumetaCompiledFunction):
                    continue

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

            lib = compiler.compile_to_library(
                name,
                obj_files,
                directory,
                libraries=libraries,
                include_dirs=include_dirs,
                libraries_dirs=libraries_dirs,
                rpath_dirs=rpath_dirs,
                additional_flags=additional_flags,
            )

            os.replace(temp_pickle_path, pickle_path)
        except Exception:
            if temp_pickle_path is not None:
                temp_pickle_path.unlink(missing_ok=True)
            raise

        return lib

    @classmethod
    def load(
        cls,
        name: str,
        directory: str | Path,
        *,
        safe: bool = False,
    ) -> "NumetaLibrary":
        cls._nm_validate_name(name)

        result = NumetaLibrary(name)

        try:
            with open(Path(directory) / f"{name}.pkl", "rb") as handle:
                for func in pickle.load(handle):
                    result._entries[func.name] = func
        except (EOFError, pickle.UnpicklingError) as exc:
            if not safe:
                raise
            warnings.warn(
                f"Failed to load NumetaLibrary '{name}' cache from {Path(directory) / f'{name}.pkl'}: {exc}. "
                "Treating it as a cache miss.",
                RuntimeWarning,
            )

        #
        #   Check collisions with already compiled function
        #

        loaded_names: set[str] = set()
        for func in result._entries.values():
            for compiled in func._compiled_functions.values():
                loaded_names.add(compiled.func_name)

        collisions = loaded_names & NumetaFunction.used_compiled_names
        if collisions:
            colliding_list = ", ".join(sorted(collisions))
            warnings.warn(
                f"Compiled function name collision while loading library '{name}'. "
                f"Existing names: {colliding_list}. "
                "Conflicting compiled entries will be dropped and rebuilt on demand.",
                RuntimeWarning,
            )
            for func in result._entries.values():
                signatures_to_drop = []
                for signature, compiled in func._compiled_functions.items():
                    if compiled.func_name in collisions:
                        signatures_to_drop.append(signature)
                for signature in signatures_to_drop:
                    func._compiled_functions.pop(signature, None)
                    func._pyc_extensions.pop(signature, None)
                    func._fast_call.pop(signature, None)

        loaded_names = set()
        for func in result._entries.values():
            for compiled in func._compiled_functions.values():
                loaded_names.add(compiled.func_name)

        if loaded_names:
            NumetaFunction.used_compiled_names.update(loaded_names)
        NumetaLibrary.loaded.add(name)

        return result
