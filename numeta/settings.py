from .syntax.settings import settings as syntax_settings


class Settings:

    def __init__(self, iso_C, use_numpy_allocator=True):
        self.iso_C = iso_C
        if self.iso_C:
            self.set_iso_C()
        else:
            self.unset_iso_C()
        self.use_numpy_allocator = use_numpy_allocator

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


settings = Settings(iso_C=True, use_numpy_allocator=True)
