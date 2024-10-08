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
        for name, _, dimension in self.fields:
            if isinstance(dimension, tuple):
                for dim in dimension:
                    if isinstance(dim, slice):
                        raise ValueError(
                            f"Dimension of structs should be defined at compile time. Got {dimension} for field {name}."
                        )
            elif isinstance(dimension, slice):
                raise ValueError(
                    f"Dimension of structs should be defined at compile time. Got {dimension} for field {name}."
                )
        self.module = None

    def get_declaration(self):
        from .statements import DerivedTypeDeclaration

        return DerivedTypeDeclaration(self)
