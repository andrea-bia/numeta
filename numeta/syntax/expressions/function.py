from .expression_node import ExpressionNode
from numeta.syntax.nodes import NamedEntity
from numeta.syntax.tools import check_node


class Function(NamedEntity, ExpressionNode):
    __slots__ = ["name", "arguments"]

    def __init__(self, name, arguments, module=None):
        super().__init__(name, module=module)
        self.arguments = [check_node(arg) for arg in arguments]

    def get_code_blocks(self):
        result = [self.name, "("]
        for argument in self.arguments:
            result.extend(argument.get_code_blocks())
            result.append(", ")
        if result[-1] == ", ":
            result.pop()
        result.append(")")
        return result

    def extract_entities(self):
        yield self
        for arg in self.arguments:
            yield from arg.extract_entities()

    def get_declaration(self):
        raise NotImplementedError("Function declaration is not supported")


# def get_with_updated_variables(self, variables_couples):
#    #new_arguments = [arg.get_with_updated_variables(variables_couples) for arg in self.arguments]
#    new_arguments = []

#    for arg in self.arguments:
#        if hasattr(arg, "get_with_updated_variables"):
#            new_arguments.append(arg.get_with_updated_variables(variables_couples))
#        else:
#            new_arguments.append(arg)

#    return Function(self.name, new_arguments)
