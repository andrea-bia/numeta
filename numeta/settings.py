from .syntax.settings import settings as syntax_settings


class Settings:

    def __init__(self, iso_C, use_numpy_allocator=True, reorder_kwargs=True):
        """Initialize the settings.
        Parameters
        ----------
        iso_C : bool
            Whether to use ISO C compatibility mode.
        use_numpy_allocator : bool
            Whether to use the NumPy memory allocator.
        reorder_kwargs : bool
            Whether to reorder keyword arguments in the generated functions.
            It permits the function to have a unique signature regardless of the order of the keyword arguments.
            But it can be a nuisance if numeta is used as code generator.
        """
        self.iso_C = iso_C
        if self.iso_C:
            self.set_iso_C()
        else:
            self.unset_iso_C()
        self.use_numpy_allocator = use_numpy_allocator
        self.__reorder_kwargs = reorder_kwargs

    @property
    def reorder_kwargs(self):
        """Return whether to reorder keyword arguments in the generated function."""
        return self.__reorder_kwargs

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

    def set_reorder_kwargs(self):
        """Set whether to reorder keyword arguments in the generated function."""
        self.__reorder_kwargs = True

    def unset_reorder_kwargs(self):
        """Unset the reordering of keyword arguments in the generated function."""
        self.__reorder_kwargs = False


settings = Settings(iso_C=True, use_numpy_allocator=True)
