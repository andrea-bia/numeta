import numpy as np

from .expression_node import ExpressionNode
from numeta.array_shape import SCALAR
from numeta.exceptions import raise_with_source


class LiteralNode(ExpressionNode):
    __slots__ = ["value", "__dtype"]

    def __init__(self, value):
        super().__init__()
        from numeta.settings import settings

        default_bool = settings.syntax.DEFAULT_BOOL
        default_int = settings.syntax.DEFAULT_INT
        default_float = settings.syntax.DEFAULT_FLOAT
        default_complex = settings.syntax.DEFAULT_COMPLEX
        default_char = settings.syntax.DEFAULT_CHAR

        if (
            default_bool is None
            or default_int is None
            or default_float is None
            or default_complex is None
            or default_char is None
        ):
            raise_with_source(
                RuntimeError,
                "Default literal datatypes are not initialized",
                source_node=self,
            )

        self.value = value
        if type(value) is bool:
            # IMPORTANT before int because bool is a subclass of int
            self.__dtype = default_bool
        elif type(value) is int:
            self.__dtype = default_int
        elif type(value) is float:
            self.__dtype = default_float
        elif type(value) is complex:
            self.__dtype = default_complex
        elif type(value) is str:
            self.__dtype = default_char
        elif isinstance(value, np.generic):
            from numeta.datatype import get_datatype

            try:
                dtype = get_datatype(value.dtype)
                if dtype is None:
                    raise ValueError("Invalid numpy scalar dtype")
                self.__dtype = dtype
            except ValueError:
                raise_with_source(
                    ValueError,
                    f"Type {value.__class__.__name__} is unsupported for LiteralNode,\n value: {value}",
                    source_node=self,
                )
        else:
            raise_with_source(
                ValueError,
                f"Type {value.__class__.__name__} is unsupported for LiteralNode,\n value: {value}",
                source_node=self,
            )

    @property
    def dtype(self):
        return self.__dtype

    @property
    def _shape(self):
        return SCALAR

    def extract_entities(self):
        dtype = self.__dtype
        if dtype is None:
            return
        ftype = dtype.get_fortran(bind_c=None)
        if ftype is not None:
            yield from ftype.extract_entities()

    def get_with_updated_variables(self, variables_couples):
        return self
