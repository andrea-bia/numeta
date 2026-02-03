from .statement import Statement
from numeta.ast.nodes import Node
from numeta.ast.settings import settings
from numeta.array_shape import UNKNOWN


class VariableDeclaration(Statement):
    def __init__(self, variable, add_to_scope=False):
        super().__init__(add_to_scope=add_to_scope)
        self.variable = variable

    def extract_entities(self):
        if getattr(self.variable, "has_ftype", True):
            yield from self.variable._ftype.extract_entities()
        elif self.variable.dtype is not None and self.variable.dtype.is_struct():
            # If it is a struct we might need to extract entities from the struct definition
            # But the struct definition is usually self contained or handled by module dependencies
            pass

        if settings.array_lower_bound != 1:
            # HACK: Non stardard array lower bound so we have to shift it
            # and well need the integer kind
            yield from settings.DEFAULT_INTEGER.extract_entities()

        if self.variable._shape is not UNKNOWN:
            for element in self.variable._shape.dims:
                if isinstance(element, Node):
                    yield from element.extract_entities()
