from .nodes import NamedEntity
from numeta.ast.nodes import NamedEntity
from .tools import check_node
from numeta.exceptions import raise_with_source


class Function(NamedEntity):
    __slots__ = ["name", "arguments", "bind_c"]

    def __init__(self, name, arguments, parent=None, bind_c=False):
        super().__init__(name, parent=parent)
        self.arguments = [check_node(arg) for arg in arguments]
        self.bind_c = bind_c

    def __call__(self, *arguments):
        from .expressions import FunctionCall

        return FunctionCall(self, *arguments)

    def extract_entities(self):
        yield self
        for arg in self.arguments:
            yield from arg.extract_entities()

    def get_declaration(self):
        raise_with_source(
            NotImplementedError, "Function declaration is not supported", source_node=self
        )

    def get_interface_declaration(self):
        from .statements import FunctionInterfaceDeclaration

        return FunctionInterfaceDeclaration(self)

    @property
    def dtype(self):
        raise_with_source(NotImplementedError, "Function dtype is not defined", source_node=self)

    @property
    def _shape(self):
        raise_with_source(NotImplementedError, "Function shape is not defined", source_node=self)

    def get_result_variable(self):
        from numeta.ast.variable import Variable

        return Variable("res0", dtype=self.dtype, shape=self._shape)

    def get_with_updated_variables(self, variables_couples):
        new_args = [arg.get_with_updated_variables(variables_couples) for arg in self.arguments]
        return type(self)(self.name, new_args, parent=self.parent, bind_c=self.bind_c)
