from .numeta_function import NumetaFunction
import pickle


def aot(directory, do_checks=True, compile_flags="-O3 -march=native", backend="fortran"):
    if callable(directory):
        raise ValueError(
            "You must provide a directory to save the compiled function, not directly the function."
        )

    def decorator_wrapper(f):
        return NumetaFunction(
            f,
            directory=directory,
            do_checks=do_checks,
            compile_flags=compile_flags,
            backend=backend,
        )

    return decorator_wrapper
