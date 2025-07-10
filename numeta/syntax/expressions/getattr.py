from .expression_node import ExpressionNode


class GetAttr(ExpressionNode):
    def __init__(self, variable, attr):
        self.variable = variable
        self.attr = attr

    @property
    def _ftype(self):
        derived_type = self.variable._ftype.kind
        for name, fortran_type, _ in derived_type.fields:
            if name == self.attr:
                return fortran_type
        raise ValueError(f"Attribute '{self.attr}' not found in derived type '{derived_type.name}'")

    @property
    def _shape(self):
        derived_type = self.variable._ftype.kind
        for name, _, shape in derived_type.fields:
            if name == self.attr:
                return shape
        raise ValueError(f"Attribute '{self.attr}' not found in derived type '{derived_type.name}'")

    def get_code_blocks(self):
        return [*self.variable.get_code_blocks(), "%", self.attr]

    def extract_entities(self):
        yield from self.variable.extract_entities()

    def get_with_updated_variables(self, variables_couples):
        return GetAttr(self.variable.get_with_updated_variables(variables_couples), self.attr)

    def __setitem__(self, key, value):
        """Does nothing, but allows to use variable[key] = value"""
        from numeta.syntax.statements import Assignment

        if isinstance(key, slice) and key.start is None and key.stop is None and key.step is None:
            # if the variable is assigned to itself, do nothing, needed for the += and -= operators
            if self is value:
                return
            Assignment(self, value)
        else:
            Assignment(self[key], value)
