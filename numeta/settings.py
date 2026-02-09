import shlex

from .fortran.settings import settings as syntax_settings
from .ast.settings import settings as ast_settings


class Settings:

    def __init__(
        self,
        iso_C,
        use_numpy_allocator=True,
        reorder_kwargs=True,
        add_shape_descriptors=True,
        ignore_fixed_shape_in_nested_calls=False,
        default_backend="fortran",
        default_do_checks=True,
        default_compile_flags="-O3 -march=native",
        use_c_dispatch=True,
    ):
        """Initialize the settings.
        Parameters
        ----------
        iso_C : bool
            Whether to use ISO C compatibility mode.
        use_numpy_allocator : bool
            Whether to use the NumPy memory allocator.
        reorder_kwargs : bool
            If True, keyword arguments in the generated functions are reordered to ensure
            a unique and deterministic function signature, independent of the order in
            which keywords are passed. This is helpful for reproducibility but may be
            inconvenient when using numeta as a code generator.
        add_shape_descriptors : bool
            If True, shape descriptors are added to array arguments in the generated
            functions. Shape descriptors encode the dimensions of arrays, which is
            typically required when using JIT compilation. However, this can be
            undesirable if numeta is used purely as a code generator.
        ignore_fixed_shape_in_nested_calls : bool
            If True, instead of passing array in a nested call as fixed shape,
            they are passed with undefined dimensions.
            This can help limiting the number of generated functions if they have no dependence on the fixed shape.
        use_c_dispatch : bool
            If True (default), use the optimized C extension for argument parsing and dispatch
            if available. If False, force the use of the pure Python implementation.
        """
        self.iso_C = iso_C
        if self.iso_C:
            self.set_iso_C()
        else:
            self.unset_iso_C()
        self.use_numpy_allocator = use_numpy_allocator
        self.__reorder_kwargs = reorder_kwargs
        self.__add_shape_descriptors = add_shape_descriptors
        self.__ignore_fixed_shape_in_nested_calls = ignore_fixed_shape_in_nested_calls
        self.set_default_backend(default_backend)
        self.set_default_do_checks(default_do_checks)
        self.set_default_compile_flags(default_compile_flags)
        self.use_c_dispatch = use_c_dispatch

    @staticmethod
    def _normalize_compile_flags(compile_flags):
        if isinstance(compile_flags, str):
            return tuple(shlex.split(compile_flags))
        return tuple(compile_flags)

    def set_default_from_datatype(self, dtype, *, iso_c: bool = False):
        """Set the default Fortran type using a :class:`DataType` subclass."""
        from .datatype import DataType

        if not isinstance(dtype, type) or not issubclass(dtype, DataType):
            raise TypeError("dtype must be a DataType subclass")

        syntax_settings.set_default_datatype(dtype)
        ast_settings.set_default_datatype(dtype)

    def set_iso_C(self):
        """Set the ISO C compatibility mode."""
        self.iso_C = True
        syntax_settings.set_c_like()
        ast_settings.set_c_like()
        from .datatype import (
            int64,
            float64,
            complex128,
            bool8,
            char,
        )

        self.set_default_from_datatype(int64, iso_c=True)
        self.set_default_from_datatype(float64, iso_c=True)
        self.set_default_from_datatype(complex128, iso_c=True)
        self.set_default_from_datatype(bool8, iso_c=True)

        self.set_default_from_datatype(char, iso_c=True)

    def unset_iso_C(self):
        """Unset the ISO C compatibility mode."""
        self.iso_C = False
        syntax_settings.unset_c_like()
        ast_settings.unset_c_like()
        from .datatype import (
            int64,
            float64,
            complex128,
            bool8,
            char,
        )

        self.set_default_from_datatype(int64, iso_c=False)
        self.set_default_from_datatype(float64, iso_c=False)
        self.set_default_from_datatype(complex128, iso_c=False)
        self.set_default_from_datatype(bool8, iso_c=False)
        self.set_default_from_datatype(char, iso_c=False)

    def set_numpy_allocator(self):
        """Set whether to use the NumPy memory allocator."""
        self.use_numpy_allocator = True

    def unset_numpy_allocator(self):
        """Unset the NumPy memory allocator."""
        self.use_numpy_allocator = False

    @property
    def reorder_kwargs(self):
        """Return whether to reorder keyword arguments in the generated function."""
        return self.__reorder_kwargs

    def set_reorder_kwargs(self):
        """Set whether to reorder keyword arguments in the generated function."""
        self.__reorder_kwargs = True

    def unset_reorder_kwargs(self):
        """Unset the reordering of keyword arguments in the generated function."""
        self.__reorder_kwargs = False

    @property
    def add_shape_descriptors(self):
        """Return whether to add shape descriptors to array arguments in generated functions."""
        return self.__add_shape_descriptors

    def set_add_shape_descriptors(self):
        """Set whether to add shape descriptors to array arguments in generated functions."""
        self.__add_shape_descriptors = True

    def unset_add_shape_descriptors(self):
        """Unset the addition of shape descriptors to array arguments in generated functions."""
        self.__add_shape_descriptors = False

    @property
    def ignore_fixed_shape_in_nested_calls(self):
        return self.__ignore_fixed_shape_in_nested_calls

    @ignore_fixed_shape_in_nested_calls.setter
    def ignore_fixed_shape_in_nested_calls(self, value):
        self.__ignore_fixed_shape_in_nested_calls = value

    @property
    def default_backend(self):
        return self.__default_backend

    def set_default_backend(self, backend: str):
        if not isinstance(backend, str):
            raise TypeError("backend must be a string")
        backend = backend.lower()
        if backend not in {"fortran", "c"}:
            raise ValueError("backend must be 'fortran' or 'c'")
        self.__default_backend = backend

    @property
    def default_do_checks(self):
        return self.__default_do_checks

    def set_default_do_checks(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("default_do_checks must be a bool")
        self.__default_do_checks = value

    @property
    def default_compile_flags(self):
        return self.__default_compile_flags

    def set_default_compile_flags(self, compile_flags):
        if compile_flags is None:
            raise TypeError("compile_flags cannot be None")
        normalized = self._normalize_compile_flags(compile_flags)
        self.__default_compile_flags = normalized

    @property
    def use_c_dispatch(self):
        return self.__use_c_dispatch

    @use_c_dispatch.setter
    def use_c_dispatch(self, value):
        if not isinstance(value, bool):
            raise TypeError("use_c_dispatch must be a bool")
        self.__use_c_dispatch = value


settings = Settings(iso_C=True, use_numpy_allocator=True)
