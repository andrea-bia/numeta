import numpy as np
import sys

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
    __slots__ = "name", "_entries"
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

        # We need to compiled ALL the NumetaFunctions not only the one directly owned by the library

        class RewritingPickler(pickle.Pickler):

            @staticmethod
            def reducer_override(obj):
                nonlocal dependencies
                nonlocal obj_files
                if isinstance(obj, NumetaFunction):
                    state = obj.__dict__.copy()
                    state["_func"] = None  # It is not pickable
                    state["_fast_call"] = {}
                    state["_pyc_extensions"] = {sig: pyc_extension for sig in obj._pyc_extensions}
                    return (NumetaFunction.__new__, (NumetaFunction,), state)

                if isinstance(obj, NumetaCompiledFunction):
                    state = obj.__dict__.copy()
                    obj_files.update([obj for obj in obj.obj_files])
                    dependencies |= obj.symbolic_function.get_dependencies()

                    state["name"] = name
                    state["_path"] = directory
                    state["_rpath"] = directory
                    state["_include"] = directory
                    state["_obj_files"] = None
                    state["compiled"] = True
                    return (NumetaCompiledFunction.__new__, (NumetaCompiledFunction,), state)
                return NotImplemented

        with open(directory / f"{self.name}.pkl", "wb") as f:
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

        return lib

    @classmethod
    def load(
        cls,
        name: str,
        directory: str | Path,
    ) -> "NumetaLibrary":
        cls._nm_validate_name(name)

        result = NumetaLibrary(name)

        with open(Path(directory) / f"{name}.pkl", "rb") as handle:
            for func in pickle.load(handle):
                result._entries[func.name] = func

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
