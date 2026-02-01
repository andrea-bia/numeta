from .nodes import Node, NamedEntity


class FortranType(Node):
    def __init__(self, type_, kind) -> None:
        super().__init__()
        self.type = type_
        self.kind = kind
        self.parent = None

    def get_code_blocks(self):
        return [self.type, "(", self.get_kind_str(), ")"]

    def extract_entities(self):
        if isinstance(self.kind, NamedEntity):
            yield self.kind

    def get_with_updated_variables(self, variables_couples):
        return self

    def get_kind_str(self):
        if isinstance(self.kind, NamedEntity):
            return self.kind.name
        return str(self.kind)

    def __eq__(self, other):
        if not isinstance(other, FortranType):
            return False
        return self.type == other.type and self._kind_key() == other._kind_key()

    def __hash__(self):
        return hash((self.type, self._kind_key()))

    def _kind_key(self):
        if isinstance(self.kind, NamedEntity):
            return ("named", self.kind.name)
        return ("value", self.kind)
