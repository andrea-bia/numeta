from .syntax.settings import settings as syntax_settings


class Settings:

    def __init__(
        self,
        iso_C,
        use_numpy_allocator=True,
        reorder_kwargs=True,
        add_shape_descriptors=True,
        ignore_fixed_shape_in_nested_calls=False,
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

    def set_default_from_datatype(self, dtype, *, iso_c: bool = False):
        """Set the default Fortran type using a :class:`DataType` subclass."""
        from .datatype import DataType

        if not isinstance(dtype, type) or not issubclass(dtype, DataType):
            raise TypeError("dtype must be a DataType subclass")

        ftype = dtype.get_fortran(bind_c=iso_c)
        syntax_settings.set_default_fortran_type(ftype)

    def set_iso_C(self):
        """Set the ISO C compatibility mode."""
        self.iso_C = True
        syntax_settings.set_c_like()
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


settings = Settings(iso_C=True, use_numpy_allocator=True)
