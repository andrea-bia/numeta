from .nodes import Node, NamedEntity


class FortranType(Node):
    def __init__(self, type_, kind) -> None:
        super().__init__()
        self.type = type_
        self.kind = kind
        self.module = None

    def get_code_blocks(self):
        if isinstance(self.kind, NamedEntity):
            return [self.type, "(", self.kind.name, ")"]
        return [self.type, "(", str(self.kind), ")"]

    def extract_entities(self):
        if isinstance(self.kind, NamedEntity):
            yield self.kind

    def get_with_updated_variables(self, variables_couples):
        return self

    def get_kind_spec(self):
        if isinstance(self.kind, NamedEntity):
            return self.kind.name
        return str(self.kind)
