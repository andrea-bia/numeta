from .statement import StatementWithScope
from .various import Comment, Import, TypingPolicy, InterfaceBlock
from .tools import (
    get_nested_dependencies_or_declarations,
    divide_variables_and_struct_types,
)


class ProcedureDeclaration(StatementWithScope):
    def __init__(self, procedure):
        self.procedure = procedure

        # First check the arguments dependencies
        entities = list(self.procedure.arguments.values())
        dependencies, declarations = get_nested_dependencies_or_declarations(
            entities, self.procedure.parent
        )
        (
            self.variables_dec,
            self.struct_types_dec,
            procedure_decs,
        ) = divide_variables_and_struct_types(declarations)
        self.interfaces = {dec.procedure.name: dec.procedure for dec in procedure_decs.values()}

        # Then check the dependencies in the body
        entities = []
        for statement in self.procedure.scope.get_statements():
            for var in statement.extract_entities():
                if var not in entities:
                    entities.append(var)
        body_dependencies, body_declarations = get_nested_dependencies_or_declarations(
            entities, self.procedure.parent
        )
        dependencies.update(body_dependencies)

        (
            body_variables_dec,
            body_struct_types_dec,
            body_procedure_decs,
        ) = divide_variables_and_struct_types(body_declarations)
        self.local_variables = {
            name: dec.variable
            for name, dec in body_variables_dec.items()
            if name not in self.variables_dec
        }
        self.variables_dec.update(body_variables_dec)
        self.struct_types_dec.update(body_struct_types_dec)
        self.interfaces |= {
            dec.procedure.name: dec.procedure for dec in body_procedure_decs.values()
        }

        self.dependencies = {}
        self.namespaces_to_import = []

        from numeta.ast.namespace import Namespace

        for dependency, var in dependencies:
            if hasattr(var, "get_interface_declaration"):
                # Should we add the interface?
                # Only if it is not contained in a namespace, if not the namespace will take care
                if not isinstance(dependency, Namespace) or dependency.hidden:
                    self.interfaces[var.name] = var
            if isinstance(dependency, Namespace):
                if not dependency.hidden:
                    self.namespaces_to_import.append((dependency, var))
                if dependency.parent is not None:
                    self.dependencies[dependency.parent.name] = dependency.parent
            else:
                self.dependencies[dependency.name] = dependency

    @property
    def children(self):
        return []

    def extract_entities(self):
        # Assume nothing is visible outside the procedure (maybe not okay?)
        yield self.procedure

    def get_statements(self):
        if self.procedure.description is not None:
            for line in self.procedure.description.split("\n"):
                yield Comment(line, add_to_scope=False)

        for dependency, var in self.namespaces_to_import:
            yield Import(dependency, only=var, add_to_scope=False)

        yield TypingPolicy(implicit_type="none", add_to_scope=False)

        if self.interfaces:
            yield InterfaceBlock(self.interfaces.values())

        yield from self.struct_types_dec.values()

        yield from self.variables_dec.values()

        yield from self.procedure.scope.get_statements()


class ProcedureInterfaceDeclaration(StatementWithScope):
    def __init__(self, procedure):
        self.procedure = procedure

    @property
    def children(self):
        return []

    def extract_entities(self):
        # Assume nothing is visible outside the interface (maybe not okay?)
        yield self.procedure

    def get_statements(self):
        if self.procedure.description is not None:
            for line in self.procedure.description.split("\n"):
                yield Comment(line, add_to_scope=False)

        # First check the arguments dependencies
        entities = list(self.procedure.arguments.values())
        dependencies, declarations = get_nested_dependencies_or_declarations(
            entities, self.procedure.parent
        )
        variables_dec, struct_types_dec, _ = divide_variables_and_struct_types(declarations)

        # Now we can construct the procedure
        for dependency, var in dependencies:
            yield Import(dependency, only=var, add_to_scope=False)

        yield TypingPolicy(implicit_type="none", add_to_scope=False)

        yield from struct_types_dec.values()

        yield from variables_dec.values()
