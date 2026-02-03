from .statement import StatementWithScope
from .various import Comment, Use, Implicit
from .tools import get_nested_dependencies_or_declarations, divide_variables_and_derived_types


class FunctionInterfaceDeclaration(StatementWithScope):
    def __init__(self, function):
        self.function = function

    @property
    def children(self):
        return []

    def extract_entities(self):
        # Assume nothing is visible outside the interface (maybe not okay?)
        yield self.function

    def get_statements(self):
        if getattr(self.function, "description", None) is not None:
            for line in self.function.description.split("\n"):
                yield Comment(line, add_to_scope=False)

        entities = list(self.function.arguments)
        result_variable = self.function.get_result_variable()
        entities.append(result_variable)

        dependencies, declarations = get_nested_dependencies_or_declarations(
            entities, self.function.parent
        )
        variables_dec, derived_types_dec, _ = divide_variables_and_derived_types(declarations)

        for dependency, var in dependencies:
            yield Use(dependency, only=var, add_to_scope=False)

        yield Implicit(implicit_type="none", add_to_scope=False)

        yield from derived_types_dec.values()

        yield from variables_dec.values()
