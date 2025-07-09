from .scope import Scope
from .tools import check_node


def inline(function, *arguments):
    """Inline ``function`` with the given ``arguments`` into the current scope."""
    # Avoid heavy imports at module load time
    from .subroutine import Subroutine

    if isinstance(function, Subroutine):
        args = [check_node(arg) for arg in arguments]
        if len(args) != len(function.arguments):
            raise ValueError("Incorrect number of arguments for inlined subroutine")
        variables_couples = list(zip(function.arguments.values(), args))
        for stmt in function.scope.get_statements():
            Scope.add_to_current_scope(stmt.get_with_updated_variables(variables_couples))
        return

    raise TypeError("Unsupported function type for inline call")
