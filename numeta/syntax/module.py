from .nodes import NamedEntity
from .subroutine import Subroutine


class Module:
    __slots__ = (
        "name",
        "description",
        "hidden",
        "dependencies",
        "derived_types",
        "interfaces",
        "variables",
        "subroutines",
    )

    def __init__(self, name, description=None, hidden=False):
        super().__init__()
        self.name = name.lower()
        self.description = description
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
            derived_type.module = self

    def add_subroutine(self, *subroutines):
        for subroutine in subroutines:
            self.subroutines[subroutine.name] = subroutine
            subroutine.module = self

    def add_variable(self, *variables):
        for variable in variables:
            self.variables[variable.name] = variable
            variable.module = self

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


builtins_module = Module(
    "builtins", "The builtins module, to contain built-in functions or subroutines"
)


class ModuleCollection:
    def __init__(self):
        self.subroutines_dictionary = {}
        self.modules_dictionary = {}

    def get_modules(self):
        return [m for m in self.modules_dictionary.values()]

    def get_or_construct(self, *key):
        if key in self.subroutines_dictionary:
            return self.subroutines_dictionary[key]

        self.subroutines_dictionary[key] = self.construct(key)
        return self.subroutines_dictionary[key]

    def construct(self, key):
        raise NotImplementedError
