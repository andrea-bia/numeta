from .expression_node import ExpressionNode
from numeta.syntax.tools import check_node


class Re(ExpressionNode):
    def __init__(self, variable):
        self.variable = variable

    def get_code_blocks(self):
        return [*self.variable.get_code_blocks(), "%", "re"]

    def extract_entities(self):
        yield from self.variable.extract_entities()

    def get_with_updated_variables(self, variables_couples):
        return Re(self.variable.get_with_updated_variables(variables_couples))


class Im(ExpressionNode):
    def __init__(self, variable):
        self.variable = variable

    def get_code_blocks(self):
        return [*self.variable.get_code_blocks(), "%", "im"]

    def extract_entities(self):
        yield from self.variable.extract_entities()

    def get_with_updated_variables(self, variables_couples):
        return Im(self.variable.get_with_updated_variables(variables_couples))

class ArrayConstructor(ExpressionNode):
    def __init__(self, *elements):
        self.elements = [check_node(e) for e in elements]

    def get_code_blocks(self):
        result = ["["]
        for element in self.elements:
            result.extend(element.get_code_blocks())
            result.append(", ")
        if result[-1] == ", ":
            result.pop()
        result.append(")")
        return result

    def extract_entities(self):
        for e in self.elements:
            yield from e.extract_entities()

    def get_with_updated_variables(self, variables_couples):
        new_elements = [
            e.get_with_updated_variables(variables_couples)
            for e in self.elements
        ]
        return ArrayConstructor(
                new_elements
        )
