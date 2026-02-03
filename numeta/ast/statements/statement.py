from numeta.ast.nodes import Node
from numeta.ast.scope import Scope


class Statement(Node):
    """
    A simple statement that can be printed and executed within a code block.

    Methods
    -------
    extract_entities():
        Extract entities referenced within the statement that have to be defined outside.

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

    def extract_entities(self):
        """Recursively extract all entities within the statement's child nodes."""
        for child in self.children:
            yield from child.extract_entities()

    def count_statements(self):
        """Return 1 for simple statements."""
        return 1

    def get_with_updated_variables(self, variables_couples):
        new_children = [
            child.get_with_updated_variables(variables_couples) for child in self.children
        ]
        return type(self)(*new_children, add_to_scope=False)


class StatementWithScope(Statement):
    """
    A statement that introduces a new scope (e.g., if, do, subroutine).

    Methods
    -------
    extract_entities():
        Extract entities referenced in the scope.

    """

    def __init__(self, add_to_scope=True, enter_scope=True):
        super().__init__(add_to_scope=add_to_scope)
        self.scope = Scope()
        if enter_scope:
            self.scope.enter()

    @property
    def children(self):
        """Return the child nodes in the statement."""
        raise NotImplementedError(
            f"Subclass '{self.__class__.__name__}' must implement the 'children' property."
        )

    def get_statements(self):
        """Return the list of statements within the scope."""
        return self.scope.get_statements()

    def extract_entities(self):
        """Extract all the visible from outside entities within this statement and its scoped statements."""
        for child in self.children:
            yield from child.extract_entities()
        for statement in self.get_statements():
            yield from statement.extract_entities()

    def get_with_updated_variables(self, variables_couples):
        new_children = [
            child.get_with_updated_variables(variables_couples) for child in self.children
        ]
        result = type(self)(*new_children, add_to_scope=False, enter_scope=False)
        result.scope = self.scope.get_with_updated_variables(variables_couples)

        return result

    def count_statements(self):
        """Count this statement and all nested statements."""
        count = 1
        for statement in self.get_statements():
            count += statement.count_statements()
        return count

    def __enter__(self):
        """Enter the scope of the statement."""
        # Add specific enter logic if needed
        pass

    def __exit__(self, *args):
        """Exit the scope of the statement."""
        self.scope.exit()
