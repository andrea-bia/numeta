import sys

from numeta.array_shape import SCALAR

from .nodes import NamedEntity
from .procedure import Procedure
from .function import Function


class Namespace(NamedEntity):
    __slots__ = (
        "name",
        "parent",
        "description",
        "hidden",
        "dependencies",
        "struct_types",
        "interfaces",
        "variables",
        "procedures",
    )

    def __init__(self, name, description=None, hidden=False, parent=None):
        super().__init__(name, parent=parent)
        self.name = name.lower()
        self.description = description
        # hidden defines if it should be a real namespace or just a container
        self.hidden = hidden

        self.dependencies = {}
        self.struct_types = {}
        self.interfaces = {}
        self.variables = {}
        self.procedures = {}

    def __getattr__(self, name):
        if name in self.__slots__:  # pragma: no cover
            return self.__getattribute__(name)
        elif name in self.variables:
            return self.variables[name]
        elif name in self.procedures:
            return self.procedures[name]
        else:
            raise AttributeError(f"Namespace {self.name} has no attribute {name}")

    def add_struct_type(self, *struct_types):
        for struct_type in struct_types:
            self.struct_types[struct_type.name] = struct_type
            struct_type.parent = self

    def add_procedure(self, *procedures):
        for procedure in procedures:
            self.procedures[procedure.name] = procedure
            procedure.parent = self

    def add_variable(self, *variables):
        for variable in variables:
            self.variables[variable.name] = variable
            variable.parent = self

    def add_interface(self, *procedures):
        for procedure in procedures:
            self.interfaces[procedure.name] = procedure

    def get_declaration(self):
        from .statements import NamespaceDeclaration

        return NamespaceDeclaration(self)

    def get_dependencies(self):
        return self.get_declaration().dependencies


builtins_namespace = Namespace(
    "builtins", "The builtins namespace, to contain built-in functions or procedures"
)


class ExternalNamespace(Namespace):
    """
    **Note**: Only to add support for methods (for external namespaces).
    When methods will be properly implemented this should be removed
    """

    def __init__(self, name, parent, hidden=False):
        super().__init__(name, hidden=hidden, parent=parent)

    def add_method(self, name, arguments, result_=None, bind_c=False):
        """
        Because currently only procedures are supported, namespaces can only have procedures.
        But ExternalNamespace should be able to have functions as well.
        """
        namespace = self

        if result_ is None:
            # It's a procedure
            method = Procedure(name, parent=namespace, bind_c=bind_c)
            for arg in arguments:
                method.add_variable(arg)
            self.add_procedure(method)

        else:
            # TODO: Arguments are not used but it could be used to check if the arguments are correct
            python_module_name = type(self).__module__
            python_module = sys.modules.get(python_module_name)

            method = type(
                name,
                (Function,),
                {
                    # to make method pickable
                    "__module__": python_module_name,
                    "_ftype": property(lambda self: result_),
                    "_shape": property(lambda self: SCALAR),
                },
            )

            # to make method pickable
            if python_module is not None:
                setattr(python_module, name, method)

            self.procedures[name] = method(
                name,
                arguments,
                parent=namespace,
                bind_c=bind_c,
            )
