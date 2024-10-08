from .nodes import NamedEntity
from .scope import Scope
from .statements import SubroutineDeclaration, InterfaceDeclaration, Call
from .syntax_settings import settings


class Subroutine(NamedEntity):
    @classmethod
    def translate(cls, original_method):
        """Permit to translate the subroutine currently being generated."""

        def wrapper(self, *args, **kwargs):
            # Save the old scope, needed to restore it after the translation when constructing more subroutines
            old_scope = Scope.current_scope
            self.scope.enter()
            result = original_method(self, *args, **kwargs)
            self.scope.exit()
            Scope.current_scope = old_scope
            return result

        return wrapper

    def __init__(
        self,
        name,
        description=None,
        pure=False,
        elemental=False,
        to_print=True,
        module=None,
        bind_c=None,
    ):
        super().__init__(name, module=module)
        if self.module is not None:
            self.module.add_subroutine(self)
        self.description = description
        self.pure = pure
        self.elemental = elemental
        self.to_print = to_print
        self.arguments = {}  # dictionary because we need an ordered set
        self.scope = Scope()
        self.bind_c = settings.subroutine_bind_c if bind_c is None else bind_c
        self.declaration = None

    def add_variable(self, *variables, with_intent=None):
        """
        Add a variable to the subroutine. If the variable is a list, it is added as a list of variables.
        Can modify the intent of the variable if with_intent is specified.
        """
        for variable in variables:
            if type(variable) is list:
                for v in variable:
                    if with_intent is not None:
                        v.intent = with_intent
                    self.arguments[v.name] = v
            else:
                if with_intent is not None:
                    variable.intent = with_intent
                self.arguments[variable.name] = variable

    def add_to_description(self, value):
        self.description += value

    def get_external_dependencies(self):
        return self.get_declaration().external_dependencies

    def get_declaration(self):
        if self.declaration is None:
            self.declaration = SubroutineDeclaration(self)
        return self.declaration

    def print_lines(self, indent=0):
        return self.get_declaration().print_lines(indent=indent)

    def get_code(self):
        return "".join(self.print_lines())

    def get_interface_declaration(self):
        return InterfaceDeclaration(self)

    def get_interface_code(self):
        return "".join(self.get_interface_declaration().print_lines())

    def __call__(self, *arguments):
        Call(self, *arguments)
