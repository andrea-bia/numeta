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
        raise NotImplementedError

    def get_kind_spec(self):
        if isinstance(self.kind, NamedEntity):
            return self.kind.name
        return str(self.kind)


FInt32 = FortranType("integer", 4)
FInt64 = FortranType("integer", 8)
FReal32 = FortranType("real", 4)
FReal64 = FortranType("real", 8)
FComplex32 = FortranType("complex", 4)
FComplex64 = FortranType("complex", 8)
FLogical8 = FortranType("logical", 1)
FCharacter = FortranType("character", 1)
