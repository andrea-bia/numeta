from .expression_node import ExpressionNode


class GetAttr(ExpressionNode):
    def __init__(self, variable, attr):
        self.variable = variable
        self.attr = attr

    def get_code_blocks(self):
        return [*self.variable.get_code_blocks(), "%", self.attr]

    def extract_entities(self):
        yield from self.variable.extract_entities()

    def get_with_updated_variables(self, variables_couples):
        return GetAttr(self.variable.get_with_updated_variables(variables_couples), self.attr)

    def __getitem__(self, key):
        if isinstance(key, slice) and key.start is None and key.stop is None and key.step is None:
            return self
        if isinstance(key, str):
            return GetAttr(self, key)
        from .getitem import GetItem

        return GetItem(self, key)

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
