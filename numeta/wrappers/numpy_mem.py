from numeta.wrappers.external_library import ExternalLibraryWrapper
from numeta.external_modules.iso_c_binding import FPointer_c, FSizet_c


class NumpyMemLib(ExternalLibraryWrapper):
    """Library exposing Numpy allocating and deallocate functions."""

    def __init__(self, directory=None, include=None, additional_flags=None):
        super().__init__("numpy_mem", directory, include, additional_flags, to_link=False)
        self.add_method("numpy_allocate", [FPointer_c, FSizet_c], None)
        self.add_method("numpy_deallocate", [FPointer_c], None)


numpy_mem = NumpyMemLib()
