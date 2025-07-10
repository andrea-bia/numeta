from .expression_node import ExpressionNode
from numeta.syntax.tools import check_node
from numeta.array_shape import ArrayShape, UNKNOWN, SCALAR


class BinaryOperationNode(ExpressionNode):
    __slots__ = ["op", "left", "right"]

    def __init__(self, left, op, right):
        self.op = op
        # self.left = left
        # self.right = right
        self.left = check_node(left)
        self.right = check_node(right)

    @property
    def _ftype(self):
        """Return the Fortran type of the expression."""
        # This is a simplification. In reality, the type of the result
        # depends on the types of the operands and the operation.
        # For example, dividing two integers should result in a real.
        # For now, we'll just return the type of the left operand.
        return self.left._ftype

    @property
    def _shape(self):
        """Return the shape of the expression if any."""
        # This is a simplification. It doesn't handle broadcasting correctly.
        # For now, we'll just return the shape of the left operand.
        if self.left._shape is SCALAR:
            return self.right._shape
        elif self.right._shape is SCALAR:
            return self.left._shape
        else:
            return self.left._shape

    def get_code_blocks(self):
        result = ["("]
        # if hasattr(self.left, "get_code_blocks"):
        #    result += self.left.get_code_blocks()
        # else:
        #    from .literal_node import LiteralNode
        #    result += LiteralNode(self.left).get_code_blocks()
        result += self.left.get_code_blocks()
        result.append(self.op)
        # if hasattr(self.right, "get_code_blocks"):
        #    result += self.right.get_code_blocks()
        # else:
        #    from .literal_node import LiteralNode
        #    result += LiteralNode(self.right).get_code_blocks()
        result += self.right.get_code_blocks()
        result.append(")")
        return result

    def get_with_updated_variables(self, variables_couples):
        return BinaryOperationNode(
            self.left.get_with_updated_variables(variables_couples),
            self.op,
            self.right.get_with_updated_variables(variables_couples),
        )

    def extract_entities(self):
        # if hasattr(self.left, "extract_entities"):
        #    yield from self.left.extract_entities()
        # if hasattr(self.right, "extract_entities"):
        #    yield from self.right.extract_entities()
        yield from self.left.extract_entities()
        yield from self.right.extract_entities()


class BinaryOperationNodeNoPar(BinaryOperationNode):
    def get_code_blocks(self):
        return [*self.left.get_code_blocks(), self.op, *self.right.get_code_blocks()]


class EqBinaryNode(BinaryOperationNode):
    __slots__ = ["op", "left", "right"]

    def __init__(self, left, right):
        # faster than calling super().__init__(left, '.eq.', right)
        self.op = ".eq."
        self.left = check_node(left)
        self.right = check_node(right)

    def __bool__(self):
        try:
            return self.left.name == self.right.name
        except AttributeError:
            raise Warning(
                f"Do not use '==' operator for non-NamedEntity: {type(self.left)} and {type(self.right)}"
            )
        # TODO: Too slow

    # from numeta.syntax.named_entity import NamedEntity
    ##Always raise Warning except for the case when we evaluating two variables
    # if isinstance(self.left, NamedEntity) and isinstance(self.right, NamedEntity):
    #    return self.left.name == self.right.name
    # raise Warning(f"Do not use '==' operator for non-NamedEntity: {type(self.left)} and {type(self.right)}")


class NeBinaryNode(BinaryOperationNode):
    def __init__(self, left, right):
        self.op = ".ne."
        # self.left = left
        # self.right = right
        self.left = check_node(left)
        self.right = check_node(right)

    def __bool__(self):
        try:
            return self.left.name != self.right.name
        except AttributeError:
            raise Warning(
                f"Do not use '!=' operator for non-NamedEntity: {type(self.left)} and {type(self.right)}"
            )

        # TODO: Too slow

    # from numeta.syntax.named_entity import NamedEntity

    ## Always raise Warning except for the case when we evaluating two variables
    # if isinstance(self.left, NamedEntity) and isinstance(self.right, NamedEntity):
    #    return self.left.name != self.right.name

    # raise Warning(f"Do not use '!=' operator for non-NamedEntity: {type(self.left)} and {type(self.right)}")
