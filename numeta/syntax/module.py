import sys

from .nodes import NamedEntity
from .subroutine import Subroutine
from .expressions import Function


class Module(NamedEntity):
    __slots__ = (
        "name",
        "parent",
        "description",
        "hidden",
        "dependencies",
        "derived_types",
        "interfaces",
        "variables",
        "subroutines",
    )

    def __init__(self, name, description=None, hidden=False, parent=None):
        super().__init__(name, parent=parent)
        self.name = name.lower()
        self.description = description
        # hidden define if it is should be a true fortran module or just a container
        self.hidden = hidden

        self.dependencies = {}
        self.derived_types = {}
        self.interfaces = {}
        self.variables = {}
        self.subroutines = {}

    def __getattr__(self, name):
        if name in self.__slots__:  # pragma: no cover
            return self.__getattribute__(name)
        elif name in self.variables:
            return self.variables[name]
        elif name in self.subroutines:
            return self.subroutines[name]
        else:
            raise AttributeError(f"Module {self.name} has no attribute {name}")

    def add_derived_type(self, *derived_types):
        for derived_type in derived_types:
            self.derived_types[derived_type.name] = derived_type
            derived_type.parent = self

    def add_subroutine(self, *subroutines):
        for subroutine in subroutines:
            self.subroutines[subroutine.name] = subroutine
            subroutine.parent = self

    def add_variable(self, *variables):
        for variable in variables:
            self.variables[variable.name] = variable
            variable.parent = self

    def add_interface(self, *subroutines):
        for subroutine in subroutines:
            self.interfaces[subroutine.name] = subroutine

    def get_declaration(self):
        from .statements import ModuleDeclaration

        return ModuleDeclaration(self)

    def print_lines(self, indent=0):
        return self.get_declaration().print_lines(indent=indent)

    def get_code(self):
        return "".join(self.print_lines())

    def get_dependencies(self):
        return self.get_declaration().dependencies


builtins_module = Module(
    "builtins", "The builtins module, to contain built-in functions or subroutines"
)


class ExternalModule(Module):
    """
    **Note**: Only to add support for methods (for external modules).
    When methods will be properly implemented this should be removed
    """

    def __init__(self, name, parent, hidden=False):
        super().__init__(name, hidden=hidden, parent=parent)

    def add_method(self, name, arguments, result_=None, bind_c=False):
        """
        Because currently only subroutines are supported, Modules can only have subroutines.
        But ExternalModule should be able to have functions as well.
        """
        module = self

        if result_ is None:
            # It's a subroutine
            method = Subroutine(name, parent=module, bind_c=bind_c)
            for arg in arguments:
                method.add_variable(arg)
            self.add_subroutine(method)

        else:
            # TODO: Arguments are not used but it could be used to check if the arguments are correct
            def __init__(self, *args):
                from .tools import check_node

                self.name = name
                self.arguments = [check_node(arg) for arg in args]
                self.parent = module

            # to make method pickable
            python_module_name = type(self).__module__
            python_module = sys.modules.get(python_module_name)

            method = type(
                name,
                (Function,),
                {
                    # to make method pickable
                    "__module__": python_module_name,
                    "__init__": __init__,
                    "_ftype": property(lambda self: result_),
                    "_shape": property(lambda self: SCALAR),
                },
            )

            # to make method pickable
            if python_module is not None:
                setattr(python_module, name, method)

            self.subroutines[name] = method
