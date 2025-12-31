from typing import (
    Any,
    Callable,
    Optional,
    overload,
)

from .numeta_function import NumetaFunction, NumetaCompilationTarget
from .registry import clear_registered_functions, registered_functions


def jitted_functions() -> list[NumetaCompilationTarget]:
    """Return a list of all functions compiled via :func:`jit`."""
    return registered_functions()


def clear_jitted_functions() -> None:
    """Clear the registry of jitted functions."""
    clear_registered_functions()


@overload
def jit(func):
    """@jit used with no arguments."""
    ...


@overload
def jit(
    *,
    directory: Optional[str] = None,
    do_checks: bool = True,
    compile_flags: str = "-O3 -march=native",
    namer: Optional[Callable[..., str]] = None,
    inline: bool | int = False,
):
    """@jit(...) used with arguments."""
    ...


def jit(
    func: Callable[..., Any] | None = None,
    *,
    directory: Optional[str] = None,
    do_checks: bool = True,
    compile_flags: str = "-O3 -march=native",
    namer: Optional[Callable[..., str]] = None,
    inline: bool | int = False,
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
        Whether to enable compile-time argument validation.
    compile_flags
        Flags for the compiler optimization step.
    namer
        Optional callable to name the JIT-generated symbols.
    inline
        Controls inlining behavior (bool or max-stmts int).

    Returns
    -------
    NumetaFunction
    """
    if func is None:

        def decorator_wrapper(f) -> NumetaFunction:
            return NumetaFunction(
                f,
                directory=directory,
                do_checks=do_checks,
                compile_flags=compile_flags,
                namer=namer,
                inline=inline,
            )

        return decorator_wrapper
    else:
        return NumetaFunction(
            func,
            directory=directory,
            do_checks=do_checks,
            compile_flags=compile_flags,
            namer=namer,
            inline=inline,
        )
