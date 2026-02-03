import numpy as np

from .expression_node import ExpressionNode
from numeta.ast.settings import settings
from numeta.array_shape import SCALAR


class LiteralNode(ExpressionNode):
    __slots__ = ["value", "__ftype"]

    def __init__(self, value):
        self.value = value
        if isinstance(value, (bool, np.bool_)):
            # IMPORTANT before int because bool is a subclass of int
            self.__ftype = settings.DEFAULT_LOGICAL
        elif isinstance(value, (int, np.int32, np.int64)):
            self.__ftype = settings.DEFAULT_INTEGER
        elif isinstance(value, (float, np.float64, np.float32)):
            self.__ftype = settings.DEFAULT_REAL
        elif isinstance(value, (complex, np.complex64, np.complex128)):
            self.__ftype = settings.DEFAULT_COMPLEX
        elif isinstance(value, str):
            self.__ftype = settings.DEFAULT_CHARACTER
        else:
            raise ValueError(
                f"Type {value.__class__.__name__} is unsupported for LiteralNode,\n value: {value}"
            )

    @property
    def _ftype(self):
        return self.__ftype

    @property
    def _shape(self):
        return SCALAR

    def extract_entities(self):
        yield from self._ftype.extract_entities()

    def get_with_updated_variables(self, variables_couples):
        return self
