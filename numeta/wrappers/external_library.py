from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence

from numeta.ast import Variable, ExternalNamespace
from numeta.fortran.fortran_type import FortranType
from numeta.external_library import ExternalLibrary
from numeta.datatype import DataType, ArrayType
from numeta.array_shape import SCALAR


class ExternalLibraryWrapper(ExternalLibrary):
    """
    A wrapper class for external library.
    Used to convert types hint to fortran symbolic variables
    """

    __slots__ = ["methods", "namespaces"]

    def __init__(
        self,
        name: str,
        directory: str | None = None,
        include: str | None = None,
        additional_flags: str | None = None,
        to_link: bool = True,
        rpath: bool = False,
    ) -> None:
        super().__init__(name, directory, include, additional_flags, to_link=to_link, rpath=rpath)
        self.methods = ExternalNamespace(name, self, hidden=True)
        self.namespaces = {}

    def add_namespace(self, name: str, hidden: bool = False) -> None:
        self.namespaces[name] = ExternalNamespace(name, self, hidden=hidden)

    def add_method(
        self,
        name: str,
        argtypes: Sequence[Arg | DataType | ArrayType | FortranType | type],
        restype: DataType | ArrayType | FortranType | type | None,
        bind_c: bool = True,
    ) -> None:
        symbolic_arguments = []
        for i, arg in enumerate(argtypes):
            if isinstance(arg, Arg):
                symbolic_arguments.append(
                    convert_argument(
                        f"a{i}",
                        arg.hint,
                        bind_c=bind_c,
                        pass_by_value=arg.pass_by_value,
                    )
                )
            else:
                symbolic_arguments.append(convert_argument(f"a{i}", arg, bind_c=bind_c))
        return_type = None
        if restype is not None:
            return_type = convert_argument("res0", restype, bind_c=bind_c).dtype

        self.methods.add_method(name, symbolic_arguments, return_type, bind_c=bind_c)

    def __getattr__(self, name: str):
        try:
            if name in self.__slots__:
                super().__getattr__(name)
            elif name in self.methods.procedures:
                return self.methods.procedures[name]
            elif name in self.namespaces:
                return self.namespaces[name]
            else:
                raise AttributeError(f"Namespace {self.name} has no attribute {name}")
        except KeyError:
            raise AttributeError(f"ExternalLibrary object has no namespace {name}")


def convert_argument(
    name: str,
    hint: DataType | ArrayType | FortranType | type,
    bind_c: bool = True,
    pass_by_value: bool | None = None,
) -> Variable:
    if isinstance(hint, ArrayType):
        dtype = hint.dtype
        shape = hint.shape
    elif isinstance(hint, FortranType):
        dtype = DataType.from_ftype(hint)
        shape = SCALAR
    elif isinstance(hint, type) and issubclass(hint, DataType):
        dtype = hint
        shape = SCALAR
    elif isinstance(hint, type) and DataType.is_np_dtype(hint):
        dtype = DataType.from_np_dtype(hint)
        shape = SCALAR
    else:
        raise TypeError(f"Expected a numpy or numeta dtype got {type(hint).__name__}")

    return Variable(
        name,
        dtype=dtype,
        use_c_types=bind_c,
        shape=shape,
        pass_by_value=pass_by_value,
    )


@dataclass(frozen=True)
class Arg:
    hint: DataType | ArrayType | FortranType | type
    pass_by_value: bool | None = None
