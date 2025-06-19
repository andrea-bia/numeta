from numeta.syntax.external_module import ExternalLibrary, ExternalModule
from numeta.syntax import Variable, FortranType
from numeta.datatype import DataType


class ExternalLibraryWrapper(ExternalLibrary):
    """
    A wrapper class for external library.
    Used to convert types hint to fortran symbolic variables
    """

    def __init__(self, name, directory=None, include=None, additional_flags=None):
        super().__init__(name, directory, include, additional_flags)

    def add_module(self, name):
        self.modules[name] = ExternalModuleWrapper(name, library=self)

    def add_method(self, name, argtypes, restype, bind_c=True):
        symbolic_arguments = [
            convert_argument(f"a{i}", arg, bind_c=bind_c) for i, arg in enumerate(argtypes)
        ]
        return_type = None
        if restype is not None:
            return_type = convert_argument("res0", restype, bind_c=bind_c)

        ExternalLibrary.add_method(self, name, symbolic_arguments, return_type, bind_c=bind_c)


class ExternalModuleWrapper(ExternalModule):
    """
    A wrapper class for external modules.
    Used to convert types hint to fortran symbolic variables
    """

    def add_method(self, name, argtypes, restype, bind_c=True):
        symbolic_arguments = [
            convert_argument(f"a{i}", arg, bind_c=bind_c) for i, arg in enumerate(argtypes)
        ]
        return_type = None
        if restype is not None:
            return_type = convert_argument("res0", restype, bind_c=bind_c)

        ExternalModule.add_method(self, name, symbolic_arguments, return_type, bind_c=bind_c)


def convert_argument(name, hint, bind_c=True):
    dimension = None
    dtype = hint
    if isinstance(hint, tuple):
        dtype, dimension = hint

    if isinstance(dtype, FortranType):
        ftype = dtype
    elif isinstance(dtype, DataType):
        ftype = dtype.get_fortran(bind_c=bind_c)
    else:
        raise TypeError("Argument type must be DataType or FortranType")

    return Variable(name, ftype=ftype, dimension=dimension)
