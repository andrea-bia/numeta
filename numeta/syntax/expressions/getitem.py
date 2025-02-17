from .expression_node import ExpressionNode
from numeta.syntax.syntax_settings import settings


class GetItem(ExpressionNode):
    def __init__(self, variable, slice_):
        self.variable = variable
        # define if only a slice [begin : end : step] of the Variable is asked
        self.sliced = slice_

    @property
    def real(self):
        from .various import Re

        return Re(self)

    @real.setter
    def real(self, value):
        from .various import Re
        from numeta.syntax.statements import Assignment

        return Assignment(Re(self), value)

    @property
    def imag(self):
        from .various import Im

        return Im(self)

    @imag.setter
    def imag(self, value):
        from .various import Im
        from numeta.syntax.statements import Assignment

        return Assignment(Im(self), value)

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
                result += get_block(slice_.stop)
            # if (lbound := settings.array_lower_bound) != 1:
            #    result += get_block(slice_.stop + (lbound -1))
            # else:
            #    result += get_block(slice_.stop)

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
            if not self.variable.fortran_order:
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

    # def get_with_updated_variables(self, variables_couples):
    #    # if the variable is present in the first index of variables_couples return the corresponding second index
    #    # and do the same work for all the variables in self.sliced

    #    for old_variable, new_variable in variables_couples:
    #        if self.id == old_variable.id:
    #            if isinstance(new_variable, Variable):
    #                return new_variable[
    #                    update_variables(self.sliced, variables_couples)
    #                ]  # if new_variable is SliceDecorator index are summed (double __getitem__)
    #            else:
    #                raise Warning(
    #                    "can't modify index of an non-Variable derived object"
    #                )
    #    # Means that it is a new variable, no need to do double getitem
    #    return self.variable[
    #        update_variables(self.sliced, variables_couples)
    #    ]  # if new_variable is SliceDecorator index are summed (double __getitem__)

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
                if self.sliced.start is None and self.sliced.stop is None:
                    new_start = None
                    new_stop = None
                else:
                    new_start = self.sliced.start if self.sliced.start is not None else 0
                    new_start += key.start if key.start is not None else 0
                    new_start -= settings.array_lower_bound

                    new_stop = self.sliced.stop if self.sliced.stop is not None else 0
                    new_stop += key.stop if key.stop is not None else 0
                    new_stop -= settings.array_lower_bound

                if self.sliced.step is not None:
                    raise Warning("Step slicing not implemented for slice merging")

                new_key = slice(new_start, new_stop, key.step)

            else:
                new_key = self.sliced.start if self.sliced.start is not None else 0
                new_key += key
                new_key -= settings.array_lower_bound

        elif key is None:
            new_key = self.sliced

        else:
            error_str = "Variable[key] not implemented for sliced different from slice or None"
            error_str += f"\nName of the variable: {self.name}"
            error_str += f"\nVariable sliced attribute: {self.sliced}"
            error_str += f"\nkey: {key}"
            raise Warning(error_str)

        return new_key
