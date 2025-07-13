from numeta.syntax.external_module import ExternalLibrary, ExternalModule
from numeta.syntax import Variable, FortranType
from numeta.datatype import DataType, ArrayType
from numeta.array_shape import SCALAR


class ExternalLibraryWrapper(ExternalLibrary):
    """
    A wrapper class for external library.
    Used to convert types hint to fortran symbolic variables
    """

    def __init__(self, name, directory=None, include=None, additional_flags=None, to_link=True):
        super().__init__(name, directory, include, additional_flags, to_link=to_link)

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
    if isinstance(hint, ArrayType):
        dtype = hint.dtype
        ftype = dtype.get_fortran(bind_c=bind_c)
        shape = hint.shape
    elif isinstance(hint, FortranType):
        ftype = hint
        shape = SCALAR
    elif isinstance(hint, type) and issubclass(hint, DataType):
        ftype = hint.get_fortran(bind_c=bind_c)
        shape = SCALAR
    elif isinstance(hint, type) and DataType.is_np_dtype(hint):
        dtype = DataType.from_np_dtype(hint)
        ftype = dtype.get_fortran(bind_c=bind_c)
        shape = SCALAR
    else:
        raise TypeError(f"Expected a numpy or numeta dtype got {type(hint).__name__}")

    return Variable(name, ftype=ftype, shape=shape)
