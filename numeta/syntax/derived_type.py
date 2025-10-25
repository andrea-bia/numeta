from .nodes import NamedEntity


class DerivedType(NamedEntity):
    """
    A derived type. Actually used to define structs.

    Parameters
    ----------
    name : str
        The name of the derived type.
    fields : list of tuples
        The fields of the derived type, each tuple containing the name, datatype, and dimension.
    """

    def __init__(self, name, fields):
        super().__init__(name)
        self.fields = fields
        for name, _, shape in self.fields:
            if shape.has_comptime_undefined_dims():
                raise ValueError(
                    f"Derived type '{name}' cannot have compile-time undefined dimensions."
                )
        self.parent = None

    def get_declaration(self):
        from .statements import DerivedTypeDeclaration

        return DerivedTypeDeclaration(self)
