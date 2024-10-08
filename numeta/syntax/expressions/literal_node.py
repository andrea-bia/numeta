from .expression_node import ExpressionNode
from numeta.syntax.nodes import NamedEntity
from numeta.syntax.syntax_settings import settings


class LiteralNode(ExpressionNode):
    __slots__ = ["value", "type_", "kind"]

    def __init__(self, value, type_="type", kind=None):
        self.value = value
        if kind is None:
            if isinstance(value, bool):
                # IMPORTANT before int because bool is a subclass of int
                self.kind = settings.DEFAULT_LOGICAL_KIND
                self.type_ = "logical"
            elif isinstance(value, int):
                self.kind = settings.DEFAULT_INTEGER_KIND
                self.type_ = "integer"
            elif isinstance(value, float):
                self.kind = settings.DEFAULT_REAL_KIND
                self.type_ = "real"
            elif isinstance(value, complex):
                self.kind = settings.DEFAULT_COMPLEX_KIND
                self.type_ = "complex"
            elif isinstance(value, str):
                self.kind = settings.DEFAULT_CHARACTER_KIND
                self.type_ = "character"
            else:
                raise ValueError(
                    f"Unknown kind for LiteralNode: {value.__class__.__name__} value: {value}"
                )
        else:
            self.kind = kind

    def extract_entities(self):
        if isinstance(self.kind, NamedEntity):
            yield self.kind

    # def get_with_updated_variables(self, variables_couples):
    #    return self.datatype.get_with_updated_variables(variables_couples)

    def get_code_blocks(self):
        if isinstance(self.kind, NamedEntity):
            kind = self.kind.name
        else:
            kind = str(self.kind)

        if self.type_ == "type":
            return [f"{self.value}"]
        elif self.type_ == "integer":
            if self.value < 0:
                return ["(", f"{int(self.value)}_{kind}", ")"]
            return [f"{int(self.value)}_{kind}"]
        elif self.type_ == "real":
            if self.value < 0.0:
                return ["(", f"{float(self.value)}_{kind}", ")"]
            return [f"{float(self.value)}_{kind}"]
        elif self.type_ == "complex":
            return [
                "(",
                f"{self.value.real}_{kind}",
                "," f"{self.value.imag}_{kind}",
                ")",
            ]
        elif self.type_ == "logical":
            if self.value is True:
                return [f".true._{kind}"]
            else:
                return [f".false._{kind}"]
        elif self.type_ == "character":
            return [f'"{self.value}"']
        else:
            raise ValueError(f"Unknown type: {self.type_}")
