import numpy as np

from .statement import Statement
from numeta.syntax.nodes import Node
from numeta.syntax.settings import settings
from .tools import get_shape_blocks
from numeta.array_shape import SCALAR, UNKNOWN


class VariableDeclaration(Statement):
    def __init__(self, variable, add_to_scope=False):
        super().__init__(add_to_scope=add_to_scope)
        self.variable = variable

    def extract_entities(self):
        yield from self.variable._ftype.extract_entities()

        if settings.array_lower_bound != 1:
            # HACK: Non stardard array lower bound so we have to shift it
            # and well need the integer kind
            yield from settings.DEFAULT_INTEGER.extract_entities()

        if self.variable._shape is not UNKNOWN:
            for element in self.variable._shape.dims:
                if isinstance(element, Node):
                    yield from element.extract_entities()

    def get_code_blocks(self):
        result = self.variable._ftype.get_code_blocks()

        if self.variable.allocatable:
            result += [", ", "allocatable"]
            result += [", ", "dimension"]
            result += ["("] + [":", ","] * (len(self.variable._shape.dims) - 1) + [":", ")"]
        elif self.variable.pointer:
            result += [", ", "pointer"]
            if self.variable._shape is not SCALAR:
                result += [", ", "dimension"]
                result += ["("] + [":", ","] * (len(self.variable._shape.dims) - 1) + [":", ")"]
        elif self.variable._shape is UNKNOWN:
            # if is a pointer
            result += [", ", "dimension"]
            result += ["(", str(settings.array_lower_bound), ":", "*", ")"]
        elif self.variable._shape.dims:
            result += [", ", "dimension"]
            result += get_shape_blocks(
                self.variable._shape.dims, fortran_order=self.variable._shape.fortran_order
            )

        if self.variable.intent is not None:
            result += [", ", "intent", "(", self.variable.intent, ")"]

        if settings.force_value:
            if self.variable._shape is SCALAR and self.variable.intent == "in":
                result += [", ", "value"]

        if self.variable.parameter:
            result += [", ", "parameter"]

        if self.variable.target:
            # why fortran? why?
            if self.variable.pointer:
                result += [", ", "contiguous"]
            else:
                result += [", ", "target"]

        if self.variable.bind_c:
            result += [", ", "bind", "(", "C", ", ", "name=", "'", self.variable.name, "'", ")"]

        result += [" :: ", self.variable.name]

        if self.variable.assign is not None:

            from numeta.syntax.expressions import LiteralNode

            if isinstance(self.variable.assign, (int, float, complex, bool, str)):
                values = LiteralNode(self.variable.assign).get_code_blocks()
            elif isinstance(self.variable.assign, np.ndarray):
                values = []
                for v in self.variable.assign.ravel():
                    values += LiteralNode(v).get_code_blocks()
                    values.append(", ")
                values.pop()
            else:
                raise ValueError("Can only assign scalars or numpy ndarrays")

            if self.variable._shape is UNKNOWN:
                raise ValueError(
                    "Cannot assign to a variable with unknown shape. "
                    "Please specify the shape of the variable."
                )
            else:
                result += [";", " data ", self.variable.name, " / ", *values, " /"]

        return result
