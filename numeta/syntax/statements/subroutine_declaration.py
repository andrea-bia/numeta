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
            entities, self.subroutine.module
        )
        self.variables_dec, self.derived_types_dec = divide_variables_and_derived_types(
            declarations
        )

        # Then check the dependencies in the body
        entities = []
        for statement in self.subroutine.scope.get_statements():
            for var in statement.extract_entities():
                if var not in entities:
                    entities.append(var)
        body_dependencies, body_declarations = get_nested_dependencies_or_declarations(
            entities, self.subroutine.module
        )
        dependencies.update(body_dependencies)

        body_variables_dec, body_derived_types_dec = divide_variables_and_derived_types(
            body_declarations
        )
        self.variables_dec.update(body_variables_dec)
        self.derived_types_dec.update(body_derived_types_dec)

        self.external_dependencies = {}
        self.interfaces = []
        self.dependencies = []
        for dependency, var in dependencies:
            if dependency.hidden:
                if hasattr(var, "get_interface_declaration"):
                    self.interfaces.append(var)
            else:
                self.dependencies.append((dependency, var))
            if dependency.external:
                self.external_dependencies[dependency.name] = dependency

    @property
    def children(self):
        return []

    def get_statements(self):
        if self.subroutine.description is not None:
            for line in self.subroutine.description.split("\n"):
                yield Comment(line, add_to_scope=False)

        for dependency, var in self.dependencies:
            yield Use(dependency, only=var, add_to_scope=False)

        yield Implicit(implicit_type="none", add_to_scope=False)

        if self.interfaces:
            yield Interface(self.interfaces)

        yield from self.derived_types_dec.values()

        yield from self.variables_dec.values()

        yield from self.subroutine.scope.get_statements()

    def get_start_code_blocks(self):
        result = []

        if self.subroutine.pure:
            result.append("pure")
            result.append(" ")

        if self.subroutine.elemental:
            result.append("elemental")
            result.append(" ")

        result.extend(["subroutine", " ", self.subroutine.name, "("])

        for variable in self.subroutine.arguments.values():
            # TODO to remove dependency on the indent
            if variable.intent is not None:
                result.extend(variable.get_code_blocks())
                result.append(", ")

        if result[-1] == ", ":
            result.pop()
        result.append(")")

        if self.subroutine.bind_c:
            result.extend([" ", "bind(C)"])

        return result

    def get_end_code_blocks(self):
        return ["end", " ", "subroutine", " ", self.subroutine.name]


class InterfaceDeclaration(StatementWithScope):
    def __init__(self, subroutine):
        self.subroutine = subroutine

    @property
    def children(self):
        return []

    def get_statements(self):
        if self.subroutine.description is not None:
            for line in self.subroutine.description.split("\n"):
                yield Comment(line, add_to_scope=False)

        # First check the arguments dependencies
        entities = list(self.subroutine.arguments.values())
        dependencies, declarations = get_nested_dependencies_or_declarations(
            entities, self.subroutine.module
        )
        variables_dec, derived_types_dec = divide_variables_and_derived_types(declarations)

        # Now we can construct the subroutine
        for dependency, var in dependencies:
            yield Use(dependency, only=var, add_to_scope=False)

        yield Implicit(implicit_type="none", add_to_scope=False)

        yield from derived_types_dec.values()

        yield from variables_dec.values()

    def get_start_code_blocks(self):
        result = []

        if self.subroutine.pure:
            result.append("pure")
            result.append(" ")

        if self.subroutine.elemental:
            result.append("elemental")
            result.append(" ")

        result.extend(["subroutine", " ", self.subroutine.name, "("])

        for variable in self.subroutine.arguments.values():
            result.extend(variable.get_code_blocks())
            result.append(", ")

        if result[-1] == ", ":
            result.pop()
        result.append(")")

        if self.subroutine.bind_c:
            result.extend([" ", "bind(C)"])

        return result

    def get_end_code_blocks(self):
        return ["end", " ", "subroutine", " ", self.subroutine.name]
