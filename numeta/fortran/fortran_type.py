from numeta.ast.nodes import NamedEntity
from numeta.ast.nodes.base_node import Node


class FortranType(Node):
    def __init__(self, type_, kind) -> None:
        super().__init__()
        self.type = type_
        self.kind = kind
        self.parent = None
        if isinstance(kind, NamedEntity):
            self._kind_str = kind.name
            self._kind_key = ("named", self._kind_str)
        else:
            self._kind_str = str(kind)
            self._kind_key = ("value", kind)
        self._hash = hash((self.type, self._kind_key))

    def get_code_blocks(self):
        return [self.type, "(", self.get_kind_str(), ")"]

    def extract_entities(self):
        if isinstance(self.kind, NamedEntity):
            yield self.kind

    def get_with_updated_variables(self, variables_couples):
        return self

    def get_kind_str(self):
        return self._kind_str

    def __eq__(self, other):
        if not isinstance(other, FortranType):
            return False
        return self.type == other.type and self._kind_key == other._kind_key

    def __hash__(self):
        return self._hash
