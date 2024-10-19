from numeta.syntax.nodes import Node, NamedEntity
from numeta.syntax.scope import Scope
from .tools import print_block


class Statement(Node):
    """
    A simple statement that can be printed and executed within a code block.

    Methods
    -------
    extract_entities():
        Extract entities referenced within the statement.

    get_code_blocks():
        Return the code blocks representing the statement.

    print_lines(indent=0):
        Print the statement as a list of formatted strings.
    """

    def __init__(self, add_to_scope=True):
        if add_to_scope:
            Scope.add_to_current_scope(self)

    @property
    def children(self):
        """Return the child nodes of the statement."""
        raise NotImplementedError(
            f"Subclass '{self.__class__.__name__}' must implement the 'children' property."
        )

    def get_code_blocks(self):
        """Return the code blocks for this statement."""
        raise NotImplementedError(
            f"Subclass '{self.__class__.__name__}' must implement 'get_code_blocks'."
        )

    def extract_entities(self):
        """Recursively extract all entities within the statement's child nodes."""
        for child in self.children:
            yield from child.extract_entities()

    def print_lines(self, indent=0):
        """Print the statement, formatted with the given indent level."""
        return [print_block(self.get_code_blocks(), indent=indent)]

    def get_with_updated_variables(self, variables_couples):
        """Return a new statement with updated variables from the given variable mappings."""
        raise NotImplementedError(
            f"Subclass '{self.__class__.__name__}' must implement 'get_with_updated_variables'."
        )
        new_children = []
        for child in self.children:
            if hasattr(child, "get_with_updated_variables"):
                new_children.append(child.get_with_updated_variables(variables_couples))
            else:
                new_children.append(child)
        return type(self)(*new_children)


class StatementWithScope(Statement):
    """
    A statement that introduces a new scope (e.g., if, do, subroutine).

    Methods
    -------
    extract_entities():
        Extract entities referenced in the scope.

    get_start_code_blocks():
        Get the starting code blocks of the scope.

    get_statements():
        Get the list of statements within the scope.

    get_end_code_blocks():
        Get the ending code blocks of the scope.

    print_lines(indent=0):
        Print the statement, including its body, as a list of formatted strings.
    """

    def __init__(self, add_to_scope=True, enter_scope=True):
        super().__init__(add_to_scope=add_to_scope)
        self.scope = Scope()
        if enter_scope:
            self.scope.enter()

    @property
    def children(self):
        """Return the child nodes in the start code and end code block."""
        raise NotImplementedError(
            f"Subclass '{self.__class__.__name__}' must implement the 'children' property."
        )

    def get_start_code_blocks(self):
        """Return the code blocks that start this scoped statement."""
        raise NotImplementedError(
            f"Subclass '{self.__class__.__name__}' must implement 'get_start_code_blocks'."
        )

    def get_end_code_blocks(self):
        """Return the code blocks that end this scoped statement."""
        raise NotImplementedError(
            f"Subclass '{self.__class__.__name__}' must implement 'get_end_code_blocks'."
        )

    def get_statements(self):
        """Return the list of statements within the scope."""
        return self.scope.get_statements()

    def print_lines(self, indent=0):
        """Print the entire scoped statement, including all nested statements."""
        result = [print_block(self.get_start_code_blocks(), indent=indent)]
        for statement in self.get_statements():
            result.extend(statement.print_lines(indent=indent + 1))
        if len(last_statement := self.get_end_code_blocks()) > 0:
            result.append(print_block(last_statement, indent=indent))
        return result

    def extract_entities(self):
        """Extract all entities within this statement and its scoped statements."""
        for child in self.children:
            yield from child.extract_entities()
        for statement in self.get_statements():
            yield from statement.extract_entities()

    def __enter__(self):
        """Enter the scope of the statement."""
        # Add specific enter logic if needed
        pass

    def __exit__(self, *args):
        """Exit the scope of the statement."""
        self.scope.exit()
