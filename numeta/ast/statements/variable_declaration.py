from .statement import Statement
from numeta.ast.nodes import Node
from numeta.array_shape import UNKNOWN


class VariableDeclaration(Statement):
    def __init__(self, variable, add_to_scope=False):
        super().__init__(add_to_scope=add_to_scope)
        self.variable = variable

    def extract_entities(self):
        # Extract entities from the variable's dtype
        if self.variable.dtype is not None:
            ftype = self.variable.dtype.get_fortran(bind_c=None)
            if ftype is not None:
                yield from ftype.extract_entities()
            elif hasattr(self.variable.dtype, "extract_entities"):
                yield from self.variable.dtype.extract_entities()
            elif self.variable.dtype.is_struct():
                # If it is a struct we might need to extract entities from the struct definition
                # But the struct definition is usually self contained or handled by module dependencies
                pass

        if self.variable._shape is not UNKNOWN:
            for element in self.variable._shape.dims:
                if isinstance(element, Node):
                    yield from element.extract_entities()
