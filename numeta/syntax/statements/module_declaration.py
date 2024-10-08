from .statement import StatementWithScope
from .various import Comment, Use, Implicit, Contains
from .tools import (
    get_nested_dependencies_or_declarations,
    divide_variables_and_derived_types,
)


class ModuleDeclaration(StatementWithScope):
    def __init__(self, module):
        self.module = module

    def get_statements(self):
        if self.module.description is not None:
            for line in self.module.description.split("\n"):
                yield Comment(line, add_to_scope=False)

        entities = list(self.module.variables.values())
        dependencies, declarations = get_nested_dependencies_or_declarations(
            entities, self.module, for_module=True
        )
        variables_dec, derived_types_dec = divide_variables_and_derived_types(declarations)

        for dependency, variable in dependencies:
            yield Use(dependency, only=variable, add_to_scope=False)

        yield from derived_types_dec.values()

        if self.module.interfaces != {}:
            raise NotImplementedError("Interfaces are not supported yet")

        yield Implicit(implicit_type="none", add_to_scope=False)

        yield from variables_dec.values()

        yield Contains(add_to_scope=False)

        yield from self.module.subroutines.values()

    def get_start_code_blocks(self):
        return ["module", " ", self.module.name]

    def get_end_code_blocks(self):
        return ["end", " ", "module", " ", self.module.name]
