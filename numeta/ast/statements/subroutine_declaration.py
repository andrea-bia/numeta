from .statement import StatementWithScope
from .various import Comment, Use, Implicit, Interface
from .tools import (
    get_nested_dependencies_or_declarations,
    divide_variables_and_derived_types,
)


class SubroutineDeclaration(StatementWithScope):
    def __init__(self, subroutine):
        self.subroutine = subroutine

        # First check the arguments dependencies
        entities = list(self.subroutine.arguments.values())
        dependencies, declarations = get_nested_dependencies_or_declarations(
            entities, self.subroutine.parent
        )
        (
            self.variables_dec,
            self.derived_types_dec,
            subroutine_decs,
        ) = divide_variables_and_derived_types(declarations)
        self.interfaces = {dec.subroutine.name: dec.subroutine for dec in subroutine_decs.values()}

        # Then check the dependencies in the body
        entities = []
        for statement in self.subroutine.scope.get_statements():
            for var in statement.extract_entities():
                if var not in entities:
                    entities.append(var)
        body_dependencies, body_declarations = get_nested_dependencies_or_declarations(
            entities, self.subroutine.parent
        )
        dependencies.update(body_dependencies)

        (
            body_variables_dec,
            body_derived_types_dec,
            body_subroutine_decs,
        ) = divide_variables_and_derived_types(body_declarations)
        self.local_variables = {
            name: dec.variable
            for name, dec in body_variables_dec.items()
            if name not in self.variables_dec
        }
        self.variables_dec.update(body_variables_dec)
        self.derived_types_dec.update(body_derived_types_dec)
        self.interfaces |= {
            dec.subroutine.name: dec.subroutine for dec in body_subroutine_decs.values()
        }

        self.dependencies = {}
        self.modules_to_import = []

        from numeta.ast.module import Module

        for dependency, var in dependencies:
            if hasattr(var, "get_interface_declaration"):
                # Should we add the interface?
                # Only if it is not contained in a module, if not the module will take care
                if not isinstance(dependency, Module) or dependency.hidden:
                    self.interfaces[var.name] = var
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
        yield self.subroutine

    def get_statements(self):
        if self.subroutine.description is not None:
            for line in self.subroutine.description.split("\n"):
                yield Comment(line, add_to_scope=False)

        for dependency, var in self.modules_to_import:
            yield Use(dependency, only=var, add_to_scope=False)

        yield Implicit(implicit_type="none", add_to_scope=False)

        if self.interfaces:
            yield Interface(self.interfaces.values())

        yield from self.derived_types_dec.values()

        yield from self.variables_dec.values()

        yield from self.subroutine.scope.get_statements()


class InterfaceDeclaration(StatementWithScope):
    def __init__(self, subroutine):
        self.subroutine = subroutine

    @property
    def children(self):
        return []

    def extract_entities(self):
        # Assume nothing is visible outside the interface (maybe not okay?)
        yield self.subroutine

    def get_statements(self):
        if self.subroutine.description is not None:
            for line in self.subroutine.description.split("\n"):
                yield Comment(line, add_to_scope=False)

        # First check the arguments dependencies
        entities = list(self.subroutine.arguments.values())
        dependencies, declarations = get_nested_dependencies_or_declarations(
            entities, self.subroutine.parent
        )
        variables_dec, derived_types_dec, _ = divide_variables_and_derived_types(declarations)

        # Now we can construct the subroutine
        for dependency, var in dependencies:
            yield Use(dependency, only=var, add_to_scope=False)

        yield Implicit(implicit_type="none", add_to_scope=False)

        yield from derived_types_dec.values()

        yield from variables_dec.values()
