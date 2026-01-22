from pathlib import Path
import pickle
import sys
import sysconfig
import importlib.util
import numpy as np

from .numeta_function import NumetaFunction, NumetaCompiledFunction
from .pyc_extension import PyCExtension
from .compiler import Compiler


class NumetaLibrary:
    __slots__ = "name", "_entries"
    loaded = set()

    def __init__(self, name: str | None = None) -> None:
        self.name = name
        self._entries: dict = {}

    def _nm_add(self, function: NumetaFunction) -> None:
        self._entries[function.name] = function

    def _nm_get(self, name) -> NumetaFunction:
        return self._entries.get(name)

    def __getattr__(self, name) -> NumetaFunction:
        if name in self.__slots__:
            super().__getattr__(name)
        elif name in self._entries:
            return self._entries[name]
        else:
            raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")

    def print_f90_files(self, directory: str | Path) -> None:
        directory = Path(directory)
        for nm_function in self._entries.values():
            for compiled_target in nm_function._compiled_targets.values():
                fortran_src = directory / f"{compiled_target.name}_src.f90"
                fortran_src.write_text(compiled_target.symbolic_function.get_code())

    def save(
        self,
        directory: str | Path,
        compile_flags: str = "",
    ) -> Path:
        directory = Path(directory).absolute()
        directory.mkdir(parents=True, exist_ok=True)

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

        compiler = Compiler("gcc", compile_flags=compile_flags)

        obj_files: set[Path] = set()
        dependencies = {}

        # We need to compiled ALL the NumetaFunctions not only the one directly owned by the library

        class RewritingPickler(pickle.Pickler):

            def reducer_override(self, obj):
                nonlocal dependencies
                nonlocal obj_files
                if isinstance(obj, NumetaFunction):
                    state = obj.__dict__.copy()
                    state["_func"] = None  # It is not pickable
                    state["_fast_call"] = {}
                    state["_pyc_extensions"] = {sig: pyc_extension for sig in obj._pyc_extensions}
                    return (NumetaFunction.__new__, (NumetaFunction,), state)

                elif isinstance(obj, NumetaCompiledFunction):
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
            self.name,
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
    ) -> None:

        if name in NumetaLibrary.loaded:
            raise ValueError(f"Already using a library called {name}")
        NumetaLibrary.loaded.add(name)

        result = NumetaLibrary(name)
        with open(Path(directory) / f"{name}.pkl", "rb") as handle:
            for func in pickle.load(handle):
                result._entries[func.name] = func

        return result
