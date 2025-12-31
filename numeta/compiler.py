from __future__ import annotations

from pathlib import Path
import subprocess as sp
import textwrap
from typing import Iterable


class FortranCompiler:
    def __init__(self, compile_flags: str | Iterable[str]) -> None:
        self.compile_flags = self._normalize_flags(compile_flags)

    @staticmethod
    def _normalize_flags(compile_flags: str | Iterable[str]) -> list[str]:
        if isinstance(compile_flags, str):
            return compile_flags.split()
        return list(compile_flags)

    def build_fortran_command(
        self,
        *,
        obj_file: Path,
        source_file: Path,
        include_dirs: Iterable[str],
        additional_flags: Iterable[str],
    ) -> list[str]:
        command = ["gfortran"]
        command.extend(["-fopenmp"])
        command.extend(self.compile_flags)
        command.extend(["-fPIC", "-c", "-o", str(obj_file)])
        command.append(str(source_file))
        command.extend([f"-I{inc_dir}" for inc_dir in include_dirs])
        command.extend(additional_flags)
        return command

    def run_command(self, command: list[str], cwd: Path) -> sp.CompletedProcess[str]:
        sp_run = sp.run(
            command,
            cwd=cwd,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            text=True,
        )
        if sp_run.returncode != 0:
            error_message = "Error while compiling, the command was:\n"
            error_message += " ".join(command) + "\n"
            error_message += "The output was:\n"
            error_message += textwrap.indent(sp_run.stdout, "    ")
            error_message += textwrap.indent(sp_run.stderr, "    ")
            raise Warning(error_message)
        return sp_run

    def compile_fortran(
        self,
        *,
        name: str,
        directory: Path,
        source: str,
        dependencies: Iterable[object],
    ) -> tuple[Path, str]:
        """
        Compile Fortran source files using gfortran and return the resulting object file.
        """
        fortran_src = directory / f"{name}_src.f90"
        fortran_src.write_text(source)

        obj_file = directory / f"{name}_fortran.o"

        include_dirs: list[str] = []
        additional_flags: list[str] = []

        for lib in dependencies:
            if lib.include is not None:
                if isinstance(lib.include, (list, tuple, set)):
                    include_dirs.extend(lib.include)
                else:
                    include_dirs.append(lib.include)
            if lib.additional_flags is not None:
                if isinstance(lib.additional_flags, str):
                    additional_flags.extend(lib.additional_flags.split())
                else:
                    additional_flags.append(lib.additional_flags)

        command = self.build_fortran_command(
            obj_file=obj_file,
            source_file=fortran_src,
            include_dirs=include_dirs,
            additional_flags=additional_flags,
        )

        self.run_command(command, cwd=directory)

        return obj_file, str(directory)
