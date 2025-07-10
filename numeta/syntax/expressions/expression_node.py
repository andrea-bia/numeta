from numeta.syntax.nodes import Node
from abc import abstractmethod


class ExpressionNode(Node):
    __slots__ = []

    def __init__(self):
        pass

    @property
    @abstractmethod
    def _ftype(self):
        """Return the Fortran type of the expression."""

    @property
    @abstractmethod
    def _shape(self):
        """Return the shape of the expression if any."""

    def _get_shape_descriptor(self):
        from .various import ArrayConstructor

        return ArrayConstructor(*self._shape.dims)

    @abstractmethod
    def extract_entities(self):
        """Extract the nested entities of the expression."""

    @abstractmethod
    def get_code_blocks(self):
        """Return the code blocks that represent the expression."""

    def get_with_updated_variables(self, variables_couples):
        return self

    def __bool__(self) -> bool:
        raise Warning("Do not use 'bool' operator for expressions.")

    def __rshift__(self, other):
        from numeta.syntax.statements import Assignment

        if isinstance(other, (int, float, complex, bool, str)):
            from .literal_node import LiteralNode

            other = LiteralNode(other)
        return Assignment(self, other)

    def __neg__(self):
        from .intrinsic_functions import Neg

        return Neg(self)

    def __abs__(self):
        from .intrinsic_functions import Abs

        return Abs(self)

    def __add__(self, other):
        from .binary_operation_node import BinaryOperationNode

        return BinaryOperationNode(self, "+", other)

    def __radd__(self, other):
        from .binary_operation_node import BinaryOperationNode

        return BinaryOperationNode(other, "+", self)

    def __sub__(self, other):
        from .binary_operation_node import BinaryOperationNode

        return BinaryOperationNode(self, "-", other)

    def __rsub__(self, other):
        from .binary_operation_node import BinaryOperationNode

        return BinaryOperationNode(other, "-", self)

    def __mul__(self, other):
        from .binary_operation_node import BinaryOperationNode

        return BinaryOperationNode(self, "*", other)

    def __rmul__(self, other):
        from .binary_operation_node import BinaryOperationNode

        return BinaryOperationNode(other, "*", self)

    def __truediv__(self, other):
        from .binary_operation_node import BinaryOperationNode

        return BinaryOperationNode(self, "/", other)

    def __rtruediv__(self, other):
        from .binary_operation_node import BinaryOperationNode

        return BinaryOperationNode(other, "/", self)

    def __floordiv__(self, other):
        from .binary_operation_node import BinaryOperationNode

        return BinaryOperationNode(self, "/", other)

    def __rfloordiv__(self, other):
        from .binary_operation_node import BinaryOperationNode

        return BinaryOperationNode(other, "/", self)

    def __pow__(self, other):
        from .binary_operation_node import BinaryOperationNode

        return BinaryOperationNode(self, "**", other)

    def __rpow__(self, other):
        from .binary_operation_node import BinaryOperationNode

        return BinaryOperationNode(other, "**", self)

    def __and__(self, other):
        from .binary_operation_node import BinaryOperationNode

        return BinaryOperationNode(self, ".and.", other)

    def __or__(self, other):
        from .binary_operation_node import BinaryOperationNode

        return BinaryOperationNode(self, ".or.", other)

    def __ne__(self, other):
        from .binary_operation_node import NeBinaryNode

        return NeBinaryNode(self, other)

    def __eq__(self, other):
        from .binary_operation_node import EqBinaryNode

        return EqBinaryNode(self, other)

    def __ge__(self, other):
        from .binary_operation_node import BinaryOperationNode

        return BinaryOperationNode(self, ".ge.", other)

    def __gt__(self, other):
        from .binary_operation_node import BinaryOperationNode

        return BinaryOperationNode(self, ".gt.", other)

    def __le__(self, other):
        from .binary_operation_node import BinaryOperationNode

        return BinaryOperationNode(self, ".le.", other)

    def __lt__(self, other):
        from .binary_operation_node import BinaryOperationNode

        return BinaryOperationNode(self, ".lt.", other)

    @property
    def real(self):
        from .various import Re

        return Re(self)

    @property
    def imag(self):
        from .various import Im

        return Im(self)

    @property
    def T(self):
        from .intrinsic_functions import Transpose

        return Transpose(self)

    def __getitem__(self, key):
        if isinstance(key, slice) and key.start is None and key.stop is None and key.step is None:
            return self
        from .getitem import GetItem
        from .getattr import GetAttr

        if isinstance(key, str):
            return GetAttr(self, key)
        return GetItem(self, key)
