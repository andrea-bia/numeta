from .expression_node import ExpressionNode
from numeta.array_shape import ArrayShape, UNKNOWN, SCALAR
from numeta.exceptions import NumetaNotImplementedError, raise_with_source
from numeta.indexing import get_slice_dim, merge_slices, merge_scalar_index


class GetItem(ExpressionNode):
    def __init__(self, variable, slice_):
        super().__init__()
        self.variable = variable
        # define if only a slice [begin : end : step] of the Variable is asked
        self.sliced = slice_

    @property
    def target(self):
        return self.variable.target

    @target.setter
    def target(self, value):
        self.variable.target = value

    @property
    def dtype(self):
        return self.variable.dtype

    @property
    def _shape(self):
        dims = []
        if self.variable._shape is UNKNOWN:
            if isinstance(self.sliced, slice):
                if self.sliced.step is not None:
                    raise_with_source(
                        NotImplementedError,
                        "Step slicing not implemented for shape extraction",
                        source_node=self,
                    )
                dims.append(get_slice_dim(self.sliced, None))
            else:
                return SCALAR
        elif self.variable._shape is SCALAR:
            return SCALAR
        elif isinstance(self.sliced, tuple):
            if all(not isinstance(element, slice) for element in self.sliced):
                return SCALAR
            for i, element in enumerate(self.sliced):
                if isinstance(element, slice):
                    if element.step is not None:
                        raise_with_source(
                            NotImplementedError,
                            "Step slicing not implemented for shape extraction",
                            source_node=self,
                        )
                    dims.append(get_slice_dim(element, self.variable._shape.dim(i)))
        else:
            if isinstance(self.sliced, slice):
                if self.sliced.step is not None:
                    raise_with_source(
                        NotImplementedError,
                        "Step slicing not implemented for shape extraction",
                        source_node=self,
                    )
                dims.append(get_slice_dim(self.sliced, self.variable._shape.dim(0)))
            else:
                return SCALAR

        return ArrayShape(tuple(dims))

    def extract_entities(self):
        yield from self.variable.extract_entities()
        from numeta.ast.tools import extract_entities

        yield from extract_entities(self.sliced)

    def __setitem__(self, key, value):
        from numeta.ast.statements import Assignment

        Assignment(self[key], value)

    def get_with_updated_variables(self, variables_couples):

        from numeta.ast.tools import update_variables

        new_var = self.variable.get_with_updated_variables(variables_couples)
        new_slice = update_variables(self.sliced, variables_couples)

        # If the variable was replaced by another GetItem (e.g. during inlining),
        # compose the indexing using the standard __getitem__ logic so that
        # slices are merged correctly.
        if isinstance(new_var, GetItem):
            return new_var[new_slice]

        # If the variable is an ArrayConstructor and the slice is an int,
        # HACK because fortran does not treat temporary variables as first class citizens
        # only for ArrayConstructor
        from .various import ArrayConstructor

        if isinstance(new_var, ArrayConstructor) and isinstance(self.sliced, int):
            return new_var.elements[self.sliced]

        return GetItem(new_var, new_slice)

    def __getitem__(self, key):
        if isinstance(key, str):
            from .getattr import GetAttr

            return GetAttr(self, key)
        else:
            new_key = self.merge_slice(key)
            return GetItem(self.variable, new_key)

    def merge_slice(self, key):
        """
        Merge the slice key with the current slice
        So for example:

            a[5:10][2:4] -> a[6:8]
        """
        new_key = None

        if isinstance(self.sliced, slice):
            if key is None:
                new_key = self.sliced

            elif isinstance(key, slice):
                if self.sliced.step is not None or key.step is not None:
                    raise_with_source(
                        NumetaNotImplementedError,
                        "Step slicing not implemented for slice merging",
                        source_node=self,
                    )

                new_key = merge_slices(self.sliced, key)

            else:
                new_key = merge_scalar_index(self.sliced, key)

        elif key is None:
            new_key = self.sliced

        else:
            error_str = "Error in array slicing. Cannot merge old slice with new one.\n"
            error_str += f"\nName of the variable: {self.variable.name}"
            error_str += f"\nOld slice: {self.sliced}"
            error_str += f"\nNew slice: {key}"
            error_str += f"\nImpossible to merge {self.variable.name}[{self.sliced}][{key}]"
            raise_with_source(ValueError, error_str, source_node=self)

        if new_key is None:
            raise_with_source(ValueError, "Failed to merge slices", source_node=self)
        return new_key
