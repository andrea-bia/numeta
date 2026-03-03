from .expression_node import ExpressionNode
from numeta.ast.tools import check_node
from numeta.array_shape import ArrayShape
from numeta.exceptions import raise_with_source


class Re(ExpressionNode):
    def __init__(self, variable):
        super().__init__()
        self.variable = variable

    @property
    def dtype(self):
        from numeta.settings import settings

        return settings.syntax.DEFAULT_FLOAT

    @property
    def _shape(self):
        return self.variable._shape

    def extract_entities(self):
        yield from self.variable.extract_entities()

    def get_with_updated_variables(self, variables_couples):
        return Re(self.variable.get_with_updated_variables(variables_couples))


class Im(ExpressionNode):
    def __init__(self, variable):
        super().__init__()
        self.variable = variable

    @property
    def dtype(self):
        from numeta.settings import settings

        return settings.syntax.DEFAULT_FLOAT

    @property
    def _shape(self):
        return self.variable._shape

    def extract_entities(self):
        yield from self.variable.extract_entities()

    def get_with_updated_variables(self, variables_couples):
        return Im(self.variable.get_with_updated_variables(variables_couples))


class ArrayConstructor(ExpressionNode):
    def __init__(self, *elements):
        super().__init__()
        self.elements = [check_node(e) for e in elements]
        self._dtype_cache = None
        self._shape_cache = None

    @property
    def dtype(self):
        cached_dtype = self._dtype_cache
        if cached_dtype is not None:
            return cached_dtype

        if not self.elements:
            raise_with_source(
                ValueError,
                "ArrayConstructor must have at least one element",
                source_node=self,
            )
        for element in self.elements:
            if element is None:
                continue
            if hasattr(element, "dtype"):
                self._dtype_cache = element.dtype
                return self._dtype_cache
        raise_with_source(
            ValueError,
            "ArrayConstructor must have at least one typed element",
            source_node=self,
        )

    @property
    def _shape(self):
        cached_shape = self._shape_cache
        if cached_shape is not None:
            return cached_shape
        shape = ArrayShape((len(self.elements),))
        self._shape_cache = shape
        return shape

    def extract_entities(self):
        for e in self.elements:
            if e is not None:
                yield from e.extract_entities()

    def get_with_updated_variables(self, variables_couples):
        new_elements = [e.get_with_updated_variables(variables_couples) for e in self.elements]
        return ArrayConstructor(*new_elements)
