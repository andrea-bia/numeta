from .statement import StatementWithScope
from .various import Comment, Use, Implicit, Contains
from .tools import (
    get_nested_dependencies_or_declarations,
    divide_variables_and_derived_types,
)


class ModuleDeclaration(StatementWithScope):
    def __init__(self, module):
        self.module = module

        entities = list(self.module.variables.values())

        dependencies, declarations = get_nested_dependencies_or_declarations(
            entities, self.module, for_module=True
        )
        self.variables_dec, self.derived_types_dec, functions_dec = (
            divide_variables_and_derived_types(declarations)
        )

        self.interfaces = [dec.subroutine for dec in functions_dec.values()]

        self.dependencies = {}
        self.modules_to_import = []

        from numeta.syntax.module import Module

        for dependency, var in dependencies:
            if hasattr(var, "get_interface_declaration"):
                # Should we add the interface?
                # Only if it is not contained in a module, if not the module will take care
                if not isinstance(dependency, Module) or dependency.hidden:
                    self.interfaces.append(var)
            if isinstance(dependency, Module):
                if not dependency.hidden:
                    self.modules_to_import.append((dependency, var))
                if dependency.parent is not None:
                    self.dependencies[dependency.parent.name] = dependency.parent
            else:
                self.dependencies[dependency.name] = dependency

    @property
    def children(self):
        return []

    def extract_entities(self):
        # Assume nothing is visible outside the Subroutine (maybe not okay?)
        yield self.module

    def get_statements(self):
        if self.module.description is not None:
            for line in self.module.description.split("\n"):
                yield Comment(line, add_to_scope=False)

        for dependency, variable in self.modules_to_import:
            yield Use(dependency, only=variable, add_to_scope=False)

        yield from self.derived_types_dec.values()

        if self.interfaces:
            raise NotImplementedError("Interfaces are not supported yet")

        yield Implicit(implicit_type="none", add_to_scope=False)

        yield from self.variables_dec.values()

        yield Contains(add_to_scope=False)

        yield from self.module.subroutines.values()

    def get_start_code_blocks(self):
        return ["module", " ", self.module.name]

    def get_end_code_blocks(self):
        return ["end", " ", "module", " ", self.module.name]
