from numeta.ast import Variable, ExternalModule
from numeta.fortran.fortran_type import FortranType
from numeta.external_library import ExternalLibrary
from numeta.datatype import DataType, ArrayType
from numeta.array_shape import SCALAR


class ExternalLibraryWrapper(ExternalLibrary):
    """
    A wrapper class for external library.
    Used to convert types hint to fortran symbolic variables
    """

    __slots__ = ["methods", "modules"]

    def __init__(
        self, name, directory=None, include=None, additional_flags=None, to_link=True, rpath=False
    ):
        super().__init__(name, directory, include, additional_flags, to_link=to_link, rpath=rpath)
        self.methods = ExternalModule(name, self, hidden=True)
        self.modules = {}

    def add_module(self, name, hidden=False):
        self.modules[name] = ExternalModule(name, self, hidden=False)

    def add_method(self, name, argtypes, restype, bind_c=True):
        symbolic_arguments = [
            convert_argument(f"a{i}", arg, bind_c=bind_c) for i, arg in enumerate(argtypes)
        ]
        return_type = None
        if restype is not None:
            return_type = convert_argument("res0", restype, bind_c=bind_c)._ftype

        self.methods.add_method(name, symbolic_arguments, return_type, bind_c=bind_c)

    def __getattr__(self, name):
        try:
            if name in self.__slots__:
                super().__getattr__(name)
            elif name in self.methods.subroutines:
                return self.methods.subroutines[name]
            elif name in self.modules:
                return self.modules[name]
            else:
                raise AttributeError(f"Module {self.name} has no attribute {name}")
        except KeyError:
            raise AttributeError(f"ExternalLibrary object has no module {name}")


def convert_argument(name, hint, bind_c=True):
    if isinstance(hint, ArrayType):
        dtype = hint.dtype
        # ftype = dtype.get_fortran(bind_c=bind_c)
        shape = hint.shape
    elif isinstance(hint, FortranType):
        # ftype = hint
        shape = SCALAR
        return Variable(name, ftype=hint, shape=shape)
    elif isinstance(hint, type) and issubclass(hint, DataType):
        dtype = hint
        # ftype = hint.get_fortran(bind_c=bind_c)
        shape = SCALAR
    elif isinstance(hint, type) and DataType.is_np_dtype(hint):
        dtype = DataType.from_np_dtype(hint)
        # ftype = dtype.get_fortran(bind_c=bind_c)
        shape = SCALAR
    else:
        raise TypeError(f"Expected a numpy or numeta dtype got {type(hint).__name__}")

    return Variable(name, dtype=dtype, use_c_types=bind_c, shape=shape)
