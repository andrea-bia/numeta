from .statement import Statement
from numeta.ast.scope import Scope
from numeta.ast.tools import check_node


class Call(Statement):
    def __init__(
        self,
        function,
        *arguments,
        add_to_scope=True,
    ):
        super().__init__(add_to_scope=add_to_scope)
        self.function = function
        self.arguments = [check_node(arg) for arg in arguments]

    @property
    def children(self):
        """Return the child nodes of the call."""
        return [self.function] + self.arguments

    def get_with_updated_variables(self, variables_couples):
        new_arguments = [
            arg.get_with_updated_variables(variables_couples) for arg in self.arguments
        ]
        return Call(self.function, *new_arguments, add_to_scope=False)
