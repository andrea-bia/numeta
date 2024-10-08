from numeta.syntax.external_module import ExternalLibrary, ExternalModule
from numeta.syntax import Variable


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
    if isinstance(hint.flags["shape"], int):
        dimension = hint.flags["shape"]
    elif isinstance(hint.flags["shape"], tuple):
        dimension = tuple([None for _ in hint.flags["shape"]])
    elif isinstance(hint.flags["shape"], slice):
        dimension = (None,)
    return Variable(name, ftype=hint.dtype.get_fortran(bind_c=bind_c), dimension=dimension)
