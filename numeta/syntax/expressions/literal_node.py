from .expression_node import ExpressionNode
from numeta.syntax.settings import settings
from numeta.array_shape import SCALAR


class LiteralNode(ExpressionNode):
    __slots__ = ["value", "__ftype"]

    def __init__(self, value):
        self.value = value
        if isinstance(value, bool):
            # IMPORTANT before int because bool is a subclass of int
            self.__ftype = settings.DEFAULT_LOGICAL
        elif isinstance(value, int):
            self.__ftype = settings.DEFAULT_INTEGER
        elif isinstance(value, float):
            self.__ftype = settings.DEFAULT_REAL
        elif isinstance(value, complex):
            self.__ftype = settings.DEFAULT_COMPLEX
        elif isinstance(value, str):
            self.__ftype = settings.DEFAULT_CHARACTER
        else:
            raise ValueError(
                f"Unknown kind for LiteralNode: {value.__class__.__name__} value: {value}"
            )

    @property
    def _ftype(self):
        return self.__ftype

    @property
    def _shape(self):
        return SCALAR

    def extract_entities(self):
        yield from self._ftype.extract_entities()

    def get_with_updated_variables(self, variables_couples):
        return self

    def get_code_blocks(self):
        kind = self._ftype.get_kind_str()

        if self._ftype.type == "type":
            return [f"{self.value}"]
        elif self._ftype.type == "integer":
            if self.value < 0:
                return ["(", f"{int(self.value)}_{kind}", ")"]
            return [f"{int(self.value)}_{kind}"]
        elif self._ftype.type == "real":
            if self.value < 0.0:
                return ["(", f"{float(self.value)}_{kind}", ")"]
            return [f"{float(self.value)}_{kind}"]
        elif self._ftype.type == "complex":
            return [
                "(",
                f"{self.value.real}_{kind}",
                "," f"{self.value.imag}_{kind}",
                ")",
            ]
        elif self._ftype.type == "logical":
            if self.value is True:
                return [f".true._{kind}"]
            else:
                return [f".false._{kind}"]
        elif self._ftype.type == "character":
            return [f'"{self.value}"']
        else:
            raise ValueError(f"Unknown type: {self._ftype.type}")
