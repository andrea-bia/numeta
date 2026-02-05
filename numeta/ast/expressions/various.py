from .expression_node import ExpressionNode
from numeta.ast.tools import check_node
from numeta.array_shape import ArrayShape


class Re(ExpressionNode):
    def __init__(self, variable):
        self.variable = variable

    @property
    def dtype(self):
        from numeta.ast.settings import settings as ast_settings

        return ast_settings.DEFAULT_FLOAT

    @property
    def _shape(self):
        return self.variable._shape

    def extract_entities(self):
        yield from self.variable.extract_entities()

    def get_with_updated_variables(self, variables_couples):
        return Re(self.variable.get_with_updated_variables(variables_couples))


class Im(ExpressionNode):
    def __init__(self, variable):
        self.variable = variable

    @property
    def dtype(self):
        from numeta.ast.settings import settings as ast_settings

        return ast_settings.DEFAULT_FLOAT

    @property
    def _shape(self):
        return self.variable._shape

    def extract_entities(self):
        yield from self.variable.extract_entities()

    def get_with_updated_variables(self, variables_couples):
        return Im(self.variable.get_with_updated_variables(variables_couples))


class ArrayConstructor(ExpressionNode):
    def __init__(self, *elements):
        self.elements = [check_node(e) for e in elements]

    @property
    def dtype(self):
        if not self.elements:
            raise ValueError("ArrayConstructor must have at least one element")
        for element in self.elements:
            if element is None:
                continue
            if hasattr(element, "dtype"):
                return element.dtype
        raise ValueError("ArrayConstructor must have at least one typed element")

    @property
    def _shape(self):
        return ArrayShape((len(self.elements),))

    def extract_entities(self):
        for e in self.elements:
            if e is not None:
                yield from e.extract_entities()

    def get_with_updated_variables(self, variables_couples):
        new_elements = [e.get_with_updated_variables(variables_couples) for e in self.elements]
        return ArrayConstructor(*new_elements)
