from numeta.wrappers.external_library import ExternalLibraryWrapper
from numeta.fortran.external_modules.iso_c_binding import FPointer_c, FSizet_c


class NumpyMemLib(ExternalLibraryWrapper):
    """Library exposing Numpy allocating and deallocate functions."""

    def __init__(self):
        super().__init__("nm_numpy_mem", to_link=False)
        self.add_method("numpy_allocate", [FPointer_c, FSizet_c], None)
        self.add_method("numpy_deallocate", [FPointer_c], None)

    def get_code(self):
        return """        
void numpy_allocate(void **ptr, size_t *nbytes) {
    *ptr = PyDataMem_NEW(*nbytes);
}

void numpy_deallocate(void **ptr) {
    PyDataMem_FREE(*ptr);
}
"""


numpy_mem = NumpyMemLib()
