from .expression_node import ExpressionNode
from numeta.syntax.settings import settings
from numeta.array_shape import ArrayShape, UNKNOWN, SCALAR


class GetItem(ExpressionNode):
    def __init__(self, variable, slice_):
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
    def _ftype(self):
        return self.variable._ftype

    @property
    def _shape(self):
        def get_dim_slice(slice_, max_dim):
            start = slice_.start
            if start is None:
                start = settings.array_lower_bound
            stop = slice_.stop
            if stop is None and max_dim is not None:
                stop = max_dim
            if slice_.step is not None:
                raise NotImplementedError("Step slicing not implemented for shape extraction")
            if stop is None:
                return None
            return stop - start + settings.array_lower_bound

        dims = []
        if self.variable._shape is UNKNOWN:
            if isinstance(self.sliced, slice):
                dims.append(get_dim_slice(self.sliced, None))
            else:
                dims.append(1)
        elif self.variable._shape is SCALAR:
            dims.append(1)
        elif isinstance(self.sliced, tuple):
            for i, element in enumerate(self.sliced):
                if isinstance(element, slice):
                    dims.append(get_dim_slice(element, self.variable._shape.dims[i]))
                else:
                    dims.append(1)
        else:
            if isinstance(self.sliced, slice):
                dims.append(get_dim_slice(self.sliced, self.variable._shape.dims[0]))
            else:
                dims.append(1)

        return ArrayShape(tuple(dims))

    def extract_entities(self):
        yield from self.variable.extract_entities()
        from numeta.syntax.tools import extract_entities

        yield from extract_entities(self.sliced)

    def get_code_blocks(self):
        result = self.variable.get_code_blocks()

        def get_block(block):
            if hasattr(block, "get_code_blocks"):
                return block.get_code_blocks()
            else:
                return [str(block)]

        def convert_slice(slice_):
            result = []

            if slice_.start is not None:
                result += get_block(slice_.start)

            result.append(":")

            if slice_.stop is not None:
                stop = slice_.stop - 1 if settings.c_like_bounds else slice_.stop
                result += get_block(stop)

            if slice_.step is not None:
                result.append(":")
                result += get_block(slice_.step)

            return result

        result.append("(")
        if isinstance(self.sliced, tuple):
            dims = []
            if hasattr(self.sliced[0], "get_code_blocks"):
                dims.append(self.sliced[0].get_code_blocks())
            elif isinstance(self.sliced[0], slice):
                dims.append(convert_slice(self.sliced[0]))
            else:
                dims.append([str(self.sliced[0])])
            for element in self.sliced[1:]:
                if hasattr(element, "get_code_blocks"):
                    dims.append(element.get_code_blocks())
                elif isinstance(element, slice):
                    dims.append(convert_slice(element))
                else:
                    dims.append([str(element)])
            if not self.variable._shape.fortran_order:
                dims = dims[::-1]

            result += dims[0]
            for dim in dims[1:]:
                result += [",", " "]
                result += dim

        else:
            if hasattr(self.sliced, "get_code_blocks"):
                result += self.sliced.get_code_blocks()
            elif isinstance(self.sliced, slice):
                result += convert_slice(self.sliced)
            else:
                result += [str(self.sliced)]
        result.append(")")
        return result

    def __setitem__(self, key, value):
        from numeta.syntax.statements import Assignment

        Assignment(self[key], value)

    def get_with_updated_variables(self, variables_couples):

        from numeta.syntax.tools import update_variables

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
        if isinstance(self.sliced, slice):
            if key is None:
                new_key = self.sliced

            elif isinstance(key, slice):
                if self.sliced.step is not None or key.step is not None:
                    raise Warning("Step slicing not implemented for slice merging")

                lb = settings.array_lower_bound

                if self.sliced.start is None and self.sliced.stop is None:
                    new_start = key.start
                    new_stop = key.stop
                else:
                    base_start = self.sliced.start if self.sliced.start is not None else lb

                    if key.start is None:
                        new_start = self.sliced.start
                    elif self.sliced.start is None:
                        new_start = key.start
                    else:
                        new_start = base_start + key.start - lb

                    if key.stop is None:
                        new_stop = self.sliced.stop
                    elif self.sliced.start is None:
                        new_stop = key.stop
                    else:
                        new_stop = base_start + key.stop - lb

                new_key = slice(new_start, new_stop, None)

            else:
                lb = settings.array_lower_bound
                base_start = self.sliced.start if self.sliced.start is not None else lb
                new_key = base_start + key - lb

        elif key is None:
            new_key = self.sliced

        else:
            error_str = "Error in array slicing. Cannot merge old slice with new one.\n"
            error_str += f"\nName of the variable: {self.variable.name}"
            error_str += f"\nOld slice: {self.sliced}"
            error_str += f"\nNew slice: {key}"
            error_str += f"\nImpossible to merge {self.variable.name}[{self.sliced}][{key}]"
            raise Warning(error_str)

        return new_key
