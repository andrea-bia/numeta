from numeta.wrappers.external_library import ExternalLibraryWrapper
from numeta.datatype import c_ptr, size_t


class NumpyMemLib(ExternalLibraryWrapper):
    """Library exposing Numpy allocating and deallocate functions."""

    def __init__(self):
        super().__init__("nm_numpy_mem", to_link=False)
        self.add_method("numpy_allocate", [c_ptr, size_t], None)
        self.add_method("numpy_deallocate", [c_ptr], None)

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
