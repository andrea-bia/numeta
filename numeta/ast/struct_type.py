from .nodes import NamedEntity
from numeta.exceptions import raise_with_source


class StructType(NamedEntity):
    """
    A structured type. Used to define structs.

    Parameters
    ----------
    name : str
        The name of the struct type.
    fields : list of tuples
        The fields of the struct type, each tuple containing the name, datatype, and dimension.
    """

    def __init__(self, name, fields):
        super().__init__(name)
        self.fields = fields
        for name, _, shape in self.fields:
            if shape.has_comptime_undefined_dims():
                raise_with_source(
                    ValueError,
                    f"Struct type '{name}' cannot have compile-time undefined dimensions.",
                    source_node=self,
                )
        self.parent = None

    def get_declaration(self):
        from .statements import StructTypeDeclaration

        return StructTypeDeclaration(self)
