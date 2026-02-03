from .statement import StatementWithScope
from .various import Comment, Import, TypingPolicy, Section
from .tools import (
    get_nested_dependencies_or_declarations,
    divide_variables_and_struct_types,
)


class NamespaceDeclaration(StatementWithScope):
    def __init__(self, namespace):
        self.namespace = namespace

        entities = list(self.namespace.variables.values())

        dependencies, declarations = get_nested_dependencies_or_declarations(
            entities, self.namespace, for_namespace=True
        )
        self.variables_dec, self.struct_types_dec, functions_dec = (
            divide_variables_and_struct_types(declarations)
        )

        self.interfaces = [dec.procedure for dec in functions_dec.values()]

        self.dependencies = {}
        self.namespaces_to_import = []

        from numeta.ast.namespace import Namespace

        for dependency, var in dependencies:
            if hasattr(var, "get_interface_declaration"):
                # Should we add the interface?
                # Only if it is not contained in a namespace, if not the namespace will take care
                if not isinstance(dependency, Namespace) or dependency.hidden:
                    self.interfaces.append(var)
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
        yield self.namespace

    def get_statements(self):
        if self.namespace.description is not None:
            for line in self.namespace.description.split("\n"):
                yield Comment(line, add_to_scope=False)

        for dependency, variable in self.namespaces_to_import:
            yield Import(dependency, only=variable, add_to_scope=False)

        yield from self.struct_types_dec.values()

        if self.interfaces:
            raise NotImplementedError("Interfaces are not supported yet")

        yield TypingPolicy(implicit_type="none", add_to_scope=False)

        yield from self.variables_dec.values()

        yield Section(add_to_scope=False)

        for procedure in self.namespace.procedures.values():
            yield procedure.get_declaration()
