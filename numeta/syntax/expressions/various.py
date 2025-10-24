from .expression_node import ExpressionNode
from numeta.syntax.tools import check_node
from numeta.array_shape import ArrayShape
from numeta.syntax.settings import settings


class Re(ExpressionNode):
    def __init__(self, variable):
        self.variable = variable

    @property
    def _ftype(self):
        return settings.DEFAULT_REAL

    @property
    def _shape(self):
        return self.variable._shape

    def get_code_blocks(self):
        return [*self.variable.get_code_blocks(), "%", "re"]

    def extract_entities(self):
        yield from self.variable.extract_entities()

    def get_with_updated_variables(self, variables_couples):
        return Re(self.variable.get_with_updated_variables(variables_couples))


class Im(ExpressionNode):
    def __init__(self, variable):
        self.variable = variable

    @property
    def _ftype(self):
        return settings.DEFAULT_REAL

    @property
    def _shape(self):
        return self.variable.shape

    def get_code_blocks(self):
        return [*self.variable.get_code_blocks(), "%", "im"]

    def extract_entities(self):
        yield from self.variable.extract_entities()

    def get_with_updated_variables(self, variables_couples):
        return Im(self.variable.get_with_updated_variables(variables_couples))


class ArrayConstructor(ExpressionNode):
    def __init__(self, *elements):
        self.elements = [check_node(e) for e in elements]

    @property
    def _ftype(self):
        if not self.elements:
            raise ValueError("ArrayConstructor must have at least one element")
        return self.elements[0].dtype

    @property
    def _shape(self):
        return ArrayShape((len(self.elements),))

    def get_code_blocks(self):
        result = ["["]
        for element in self.elements:
            if element is None:
                result.append("None")
            else:
                result.extend(element.get_code_blocks())
            result.append(", ")
        if result[-1] == ", ":
            result.pop()
        result.append("]")
        return result

    def extract_entities(self):
        for e in self.elements:
            if e is not None:
                yield from e.extract_entities()

    def get_with_updated_variables(self, variables_couples):
        new_elements = [e.get_with_updated_variables(variables_couples) for e in self.elements]
        return ArrayConstructor(*new_elements)
