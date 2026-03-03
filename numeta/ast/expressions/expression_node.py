from abc import abstractmethod

from numeta.ast.nodes import Node
from numeta.exceptions import NumetaTypeError, raise_with_source


BinaryOperationNode = None
EqBinaryNode = None
NeBinaryNode = None
GetItem = None
GetAttr = None
Neg = None
Abs = None
Transpose = None
Re = None
Im = None
Assignment = None


class ExpressionNode(Node):
    __slots__ = []

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def dtype(self):
        """Return the DataType of the expression."""

    @property
    @abstractmethod
    def _shape(self):
        """Return the shape of the expression if any."""

    def _get_shape_descriptor(self):
        from .various import ArrayConstructor

        return ArrayConstructor(*self._shape.as_tuple())

    @abstractmethod
    def extract_entities(self):
        """Extract the nested entities of the expression."""

    def get_with_updated_variables(self, variables_couples):
        return self

    def __bool__(self) -> bool:
        raise_with_source(
            NumetaTypeError,
            "Do not use 'bool' operator for expressions.",
            source_node=self,
        )

    def __rshift__(self, other):
        if isinstance(other, (int, float, complex, bool, str)):
            from .literal_node import LiteralNode

            other = LiteralNode(other)
        from numeta.ast.statements import Assignment

        return Assignment(self, other)

    def __neg__(self):
        return Neg(self)

    def __abs__(self):
        return Abs(self)

    def __add__(self, other):
        return BinaryOperationNode(self, "+", other)

    def __radd__(self, other):
        return BinaryOperationNode(other, "+", self)

    def __sub__(self, other):
        return BinaryOperationNode(self, "-", other)

    def __rsub__(self, other):
        return BinaryOperationNode(other, "-", self)

    def __mul__(self, other):
        return BinaryOperationNode(self, "*", other)

    def __rmul__(self, other):
        return BinaryOperationNode(other, "*", self)

    def __truediv__(self, other):
        return BinaryOperationNode(self, "/", other)

    def __rtruediv__(self, other):
        return BinaryOperationNode(other, "/", self)

    def __floordiv__(self, other):
        return BinaryOperationNode(self, "/", other)

    def __rfloordiv__(self, other):
        return BinaryOperationNode(other, "/", self)

    def __pow__(self, other):
        return BinaryOperationNode(self, "**", other)

    def __rpow__(self, other):
        return BinaryOperationNode(other, "**", self)

    def __and__(self, other):
        return BinaryOperationNode(self, ".and.", other)

    def __or__(self, other):
        return BinaryOperationNode(self, ".or.", other)

    def __ne__(self, other):
        return NeBinaryNode(self, other)

    def __eq__(self, other):
        return EqBinaryNode(self, other)

    def __ge__(self, other):
        return BinaryOperationNode(self, ".ge.", other)

    def __gt__(self, other):
        return BinaryOperationNode(self, ".gt.", other)

    def __le__(self, other):
        return BinaryOperationNode(self, ".le.", other)

    def __lt__(self, other):
        return BinaryOperationNode(self, ".lt.", other)

    @property
    def real(self):
        return Re(self)

    @property
    def imag(self):
        return Im(self)

    @property
    def T(self):
        return Transpose(self)

    def __getitem__(self, key):
        if isinstance(key, slice) and key.start is None and key.stop is None and key.step is None:
            return self
        if isinstance(key, str):
            return GetAttr(self, key)
        return GetItem(self, key)


def _bind_expression_classes():
    global BinaryOperationNode, EqBinaryNode, NeBinaryNode
    global GetItem, GetAttr, Neg, Abs, Transpose, Re, Im

    from .binary_operation_node import BinaryOperationNode, EqBinaryNode, NeBinaryNode
    from .getattr import GetAttr
    from .getitem import GetItem
    from .intrinsic_functions import Abs, Neg, Transpose
    from .various import Im, Re


_bind_expression_classes()
