from .expression_node import ExpressionNode
from numeta.exceptions import raise_with_source


class GetAttr(ExpressionNode):
    def __init__(self, variable, attr):
        super().__init__()
        self.variable = variable
        self.attr = attr

    @property
    def dtype(self):
        struct_dtype = self.variable.dtype
        if hasattr(struct_dtype, "_members"):
            for name, dtype, _ in struct_dtype._members:
                if name == self.attr:
                    return dtype
        raise_with_source(
            ValueError,
            f"Attribute '{self.attr}' not found in struct type '{struct_dtype}'",
            source_node=self,
        )

    @property
    def _shape(self):
        struct_dtype = self.variable.dtype
        if hasattr(struct_dtype, "_members"):
            for name, _, shape in struct_dtype._members:
                if name == self.attr:
                    return shape
        raise_with_source(
            ValueError,
            f"Attribute '{self.attr}' not found in struct type '{struct_dtype}'",
            source_node=self,
        )

    def extract_entities(self):
        yield from self.variable.extract_entities()

    def get_with_updated_variables(self, variables_couples):
        return GetAttr(self.variable.get_with_updated_variables(variables_couples), self.attr)

    def __setitem__(self, key, value):
        """Does nothing, but allows to use variable[key] = value"""
        from numeta.ast.statements import Assignment

        if isinstance(key, slice) and key.start is None and key.stop is None and key.step is None:
            # if the variable is assigned to itself, do nothing, needed for the += and -= operators
            if self is value:
                return
            Assignment(self, value)
        else:
            Assignment(self[key], value)
