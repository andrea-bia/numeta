import numpy as np

from .expression_node import ExpressionNode
from numeta.array_shape import SCALAR
from numeta.exceptions import raise_with_source


class LiteralNode(ExpressionNode):
    __slots__ = ["value", "__dtype"]

    def __init__(self, value):
        super().__init__()
        from numeta.datatype import int64, float64, complex128, bool8, char

        self.value = value
        if isinstance(value, (bool, np.bool_)):
            # IMPORTANT before int because bool is a subclass of int
            self.__dtype = bool8
        elif isinstance(value, (int, np.int32, np.int64)):
            self.__dtype = int64
        elif isinstance(value, (float, np.float64, np.float32)):
            self.__dtype = float64
        elif isinstance(value, (complex, np.complex64, np.complex128)):
            self.__dtype = complex128
        elif isinstance(value, str):
            self.__dtype = char
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
        ftype = self.__dtype.get_fortran(bind_c=None)
        if ftype is not None:
            yield from ftype.extract_entities()

    def get_with_updated_variables(self, variables_couples):
        return self
