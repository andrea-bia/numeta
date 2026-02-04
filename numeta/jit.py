import warnings

from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    overload,
)

from .numeta_function import NumetaFunction
from .numeta_library import NumetaLibrary
from .settings import settings


@overload
def jit(func):
    """@jit used with no arguments."""
    ...


@overload
def jit(
    *,
    directory: Optional[str] = None,
    do_checks: bool | None = None,
    compile_flags: str | Iterable[str] | None = None,
    namer: Optional[Callable[..., str]] = None,
    inline: bool | int = False,
    library: NumetaLibrary | None = None,
    backend: str | None = None,
):
    """@jit(...) used with arguments."""
    ...


def jit(
    func: Callable[..., Any] | None = None,
    *,
    directory: Optional[str] = None,
    do_checks: bool | None = None,
    compile_flags: str | Iterable[str] | None = None,
    namer: Optional[Callable[..., str]] = None,
    inline: bool | int = False,
    library: NumetaLibrary | None = None,
    backend: str | None = None,
):
    """
    Compile a function with the Numeta JIT, either directly or via parameters.

    Overload Resolution
    -------------------
    1.  **No-arg form**: `@jit`
        - Returns a `NumetaFunction` wrapping the target.
    2.  **With-arg form**: `@jit(directory=..., inline=2, ...)`
        - Returns a decorator that, when applied, produces a `NumetaFunction`.

    Parameters
    ----------
    func
        The function to compile when using `@jit` with no args.
    directory
        Target directory for compiled output (default: none → temp dir).
    do_checks
        Whether to enable compile-time argument validation. If None, uses settings default.
    compile_flags
        Flags for the compiler optimization step. If None, uses settings default.
    namer
        Optional callable to name the JIT-generated symbols.
    inline
        Controls inlining behavior (bool or max-stmts int).
    library
        Optional library container used to group jitted functions.
    backend
        Backend to use for code generation ("fortran" or "c"). If None, uses settings default.

    Returns
    -------
    NumetaFunction
    """
    if backend is None:
        backend = settings.default_backend
    if do_checks is None:
        do_checks = settings.default_do_checks
    if compile_flags is None:
        compile_flags = settings.default_compile_flags
    compile_flags = settings._normalize_compile_flags(compile_flags)
    compile_flags_list = list(compile_flags)
    if func is None:

        def decorator_wrapper(f) -> NumetaFunction:
            name = f.__name__
            if name.startswith("_nm"):
                raise ValueError("Cannot create functions that startwith '_nm'")
            if library is not None and name in library:
                nm_func = library[name]
                if nm_func.do_checks != do_checks:
                    warnings.warn(
                        f"function {name} has been loaded with different do_checks value: {nm_func.do_checks}",
                        stacklevel=2,
                    )
                if tuple(nm_func.compile_flags) != tuple(compile_flags_list):
                    warnings.warn(
                        f"function {name} has been loaded with different compile_flags value: {nm_func.compile_flags}",
                        stacklevel=2,
                    )
            else:
                nm_func = NumetaFunction(
                    f,
                    directory=directory,
                    do_checks=do_checks,
                    compile_flags=compile_flags_list,
                    namer=namer,
                    inline=inline,
                    backend=backend,
                )
                if library is not None:
                    library.register(nm_func)
            return nm_func

        return decorator_wrapper
    else:
        nm_func = NumetaFunction(
            func,
            directory=directory,
            do_checks=do_checks,
            compile_flags=compile_flags_list,
            namer=namer,
            inline=inline,
            backend=backend,
        )
        return nm_func
