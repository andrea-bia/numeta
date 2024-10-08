from .expression_node import ExpressionNode


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
