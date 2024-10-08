from .statement import Statement
from numeta.syntax.scope import Scope
from numeta.syntax.tools import check_node


class Call(Statement):
    def __init__(
        self,
        function,
        *arguments,
        force_variables=False,
        inline=False,
        **inline_variables,
    ):
        Scope.add_to_current_scope(self)
        self.function = function
        self.arguments = [check_node(arg) for arg in arguments]
        self.force_variables = force_variables
        self.inline = inline
        self.inline_variables = inline_variables

    # def get_with_updated_variables(self, variables_couples):
    #    new_arguments = []
    #    for arg in self.arguments:
    #        if hasattr(arg, "get_with_updated_variables"):
    #            new_arguments.append(arg.get_with_updated_variables(variables_couples))
    #        else:
    #            new_arguments.append(arg)

    #    return Call(function, new_arguments)

    def extract_entities(self):
        yield from self.function.extract_entities()
        for arg in self.arguments:
            yield from arg.extract_entities()

    def get_code_blocks(self):
        if isinstance(self.function, str):
            print("todo function name")
            result = ["call", " ", self.function]
        else:
            result = ["call", " ", self.function.name]

        result += ["("]
        for arg in self.arguments:
            result += arg.get_code_blocks()
            result += [", "]
        if result[-1] == ", ":
            result.pop()
        result += [")"]

        return result
