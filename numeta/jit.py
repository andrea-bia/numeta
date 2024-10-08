from .numeta_function import NumetaFunction


def jit(func=None, *, directory=None, do_checks=True, compile_flags="-O3 -march=native"):
    if func is None:
        # Decorator is called with arguments
        def decorator_wrapper(f):
            return NumetaFunction(
                f, directory=directory, do_checks=do_checks, compile_flags=compile_flags
            )

        return decorator_wrapper
    else:
        # Decorator is called without arguments
        return NumetaFunction(
            func, directory=directory, do_checks=do_checks, compile_flags=compile_flags
        )
