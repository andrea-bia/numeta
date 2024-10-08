from .expressions import Function
from .subroutine import Subroutine
from .module import Module


class ExternalLibrary(Module):
    """
    A class to represent an external library.
    It is used to link external libraries to the fortran code.
    Is is child of Module class, where the module is hidden.
    Can contain ExternalModule objects.
    """

    def __init__(self, name, directory=None, include=None, additional_flags=None):
        """
        Directory is the path to the directory where the external library to link is located.
        Include is the path of the header file to include.
        """
        super().__init__(name, hidden=True)
        self.external = True
        self.directory = directory
        self.include = include
        self.additional_flags = additional_flags
        self.modules = {}

    def get_declaration(self):
        raise NotImplementedError("External libraries cannot be declared")

    def __getattr__(self, name):
        try:
            if name in self.variables:
                return self.variables[name]
            elif name in self.subroutines:
                return self.subroutines[name]
            elif name in self.modules:
                return self.modules[name]
            else:
                raise AttributeError(f"Module {self.name} has no attribute {name}")
        except KeyError:
            raise AttributeError(f"ExternalLibrary object has no module {name}")

    def add_module(self, name):
        self.modules[name] = ExternalModule(name)

    def add_method(self, name, arguments, result_=None, bind_c=False):
        """
        Because currently only subroutines are supported, Modules can only have subroutines.
        But ExternalModule should be able to have functions as well.
        """
        module = self

        if result_ is None:
            # It's a subroutine
            method = ExternalSubroutine(name, arguments, module=module, bind_c=bind_c)
            self.add_subroutine(method)

        else:
            # TODO: Arguments are not used but it could be used to check if the arguments are correct
            def __init__(self, *args):
                self.name = name
                self.arguments = args
                self.module = module

            method = type(name, (Function,), {"__init__": __init__})
            self.subroutines[name] = method


class ExternalModule(Module):
    def __init__(self, name, library=None):
        super().__init__(name)
        self.external = True
        self.library = library

    def get_declaration(self):
        raise NotImplementedError("External modules cannot be declared")

    def add_method(self, name, arguments, result_=None, bind_c=False):
        """
        Because currently only subroutines are supported, Modules can only have subroutines.
        But ExternalModule should be able to have functions as well.
        """
        module = self

        if result_ is None:
            # It's a subroutine
            method = ExternalSubroutine(name, arguments, module=module, bind_c=bind_c)
            self.add_subroutine(method)

        else:
            # TODO: Arguments are not used but it could be used to check if the arguments are correct
            def __init__(self, *args):
                self.name = name
                self.arguments = args
                self.module = module

            method = type(name, (Function,), {"__init__": __init__})
            self.subroutines[name] = method


class ExternalSubroutine(Subroutine):
    def __init__(self, name, arguments, module=None, bind_c=True):
        super().__init__(name, module=module, bind_c=bind_c)
        for arg in arguments:
            self.add_variable(arg)

    def get_declaration(self):
        raise NotImplementedError("External methods cannot be declared")
