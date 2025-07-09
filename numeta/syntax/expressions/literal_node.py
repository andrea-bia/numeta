from .expression_node import ExpressionNode
from numeta.syntax.settings import settings


class LiteralNode(ExpressionNode):
    __slots__ = ["value", "ftype"]

    def __init__(self, value):
        self.value = value
        if isinstance(value, bool):
            # IMPORTANT before int because bool is a subclass of int
            self.ftype = settings.DEFAULT_LOGICAL
        elif isinstance(value, int):
            self.ftype = settings.DEFAULT_INTEGER
        elif isinstance(value, float):
            self.ftype = settings.DEFAULT_REAL
        elif isinstance(value, complex):
            self.ftype = settings.DEFAULT_COMPLEX
        elif isinstance(value, str):
            self.ftype = settings.DEFAULT_CHARACTER
        else:
            raise ValueError(
                f"Unknown kind for LiteralNode: {value.__class__.__name__} value: {value}"
            )

    def extract_entities(self):
        yield from self.ftype.extract_entities()

    def get_with_updated_variables(self, variables_couples):
        return self

    def get_code_blocks(self):
        kind = self.ftype.get_kind_str()

        if self.ftype.type == "type":
            return [f"{self.value}"]
        elif self.ftype.type == "integer":
            if self.value < 0:
                return ["(", f"{int(self.value)}_{kind}", ")"]
            return [f"{int(self.value)}_{kind}"]
        elif self.ftype.type == "real":
            if self.value < 0.0:
                return ["(", f"{float(self.value)}_{kind}", ")"]
            return [f"{float(self.value)}_{kind}"]
        elif self.ftype.type == "complex":
            return [
                "(",
                f"{self.value.real}_{kind}",
                "," f"{self.value.imag}_{kind}",
                ")",
            ]
        elif self.ftype.type == "logical":
            if self.value is True:
                return [f".true._{kind}"]
            else:
                return [f".false._{kind}"]
        elif self.ftype.type == "character":
            return [f'"{self.value}"']
        else:
            raise ValueError(f"Unknown type: {self.ftype.type}")
