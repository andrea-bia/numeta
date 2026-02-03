from .expression_node import ExpressionNode
from numeta.ast.tools import check_node
from .expression_node import ExpressionNode


class FunctionCall(ExpressionNode):
    __slots__ = ["function", "arguments"]

    def __init__(self, function, *arguments):
        self.function = function
        self.arguments = [check_node(arg) for arg in arguments]

    @property
    def _ftype(self):
        return self.function._ftype

    @property
    def _shape(self):
        return self.function._shape

    def get_code_blocks(self):
        result = [self.function.name, "("]
        for argument in self.arguments:
            result.extend(argument.get_code_blocks())
            result.append(", ")
        if result[-1] == ", ":
            result.pop()
        result.append(")")
        return result

    def extract_entities(self):
        yield self.function
        for arg in self.arguments:
            yield from arg.extract_entities()

    def get_with_updated_variables(self, variables_couples):
        new_args = [arg.get_with_updated_variables(variables_couples) for arg in self.arguments]
        return FunctionCall(self.function, *new_args)
