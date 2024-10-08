from .statement import Statement
from numeta.syntax.nodes import NamedEntity
from numeta.syntax.syntax_settings import settings


class VariableDeclaration(Statement):
    def __init__(self, variable, add_to_scope=False):
        super().__init__(add_to_scope=add_to_scope)
        self.variable = variable

    def extract_entities(self):
        yield from self.variable.ftype.extract_entities()

        if settings.array_lower_bound != 1:
            # Non stardard array lower bound so we have to shift it
            # and well need the integer kind
            if isinstance(settings.DEFAULT_INTEGER_KIND, NamedEntity):
                yield settings.DEFAULT_INTEGER_KIND

        if isinstance(self.variable.dimension, NamedEntity):
            # TODO it is not okay
            yield self.variable.dimension
        elif isinstance(self.variable.dimension, tuple):
            for element in self.variable.dimension:
                if isinstance(element, NamedEntity):
                    yield element

    def get_code_blocks(self):
        result = self.variable.ftype.get_code_blocks()

        if self.variable.allocatable:
            result += [", ", "allocatable"]
            result += [", ", "dimension", "("]
            result += [":", ", "] * (len(self.variable.dimension) - 1)
            result += [":", ")"]
        elif self.variable.dimension is not None:
            result += [", ", "dimension", "("]

            def convert(element):
                if element is None:
                    if settings.array_lower_bound != 1:
                        return ["0", ":", "*"]
                    else:
                        return ["*"]
                elif isinstance(element, int):
                    if (lbound := settings.array_lower_bound) != 1:
                        return [str(lbound), ":", str(element + (lbound - 1))]
                    else:
                        return [str(element)]
                elif isinstance(element, str):
                    # TODO: maybe to remove this
                    return [element]
                elif isinstance(element, slice):
                    if element.start is None:
                        if (lbound := settings.array_lower_bound) != 1:
                            start = [str(lbound)]
                        else:
                            start = ["1"]
                    elif isinstance(element.start, int):
                        start = [str(element.start)]
                    else:
                        start = element.start.get_code_blocks()

                    if element.stop is None:
                        stop = [""]
                    elif isinstance(element.stop, int):
                        stop = [str(element.stop)]
                    else:
                        stop = element.stop.get_code_blocks()

                    if element.step is not None:
                        raise NotImplementedError("Step in array dimensions is not implemented yet")
                    return start + [":"] + stop
                else:
                    if (lbound := settings.array_lower_bound) != 1:
                        return [
                            str(lbound),
                            ":",
                            *(element + (lbound - 1)).get_code_blocks(),
                        ]
                    else:
                        return element.get_code_blocks()

            if isinstance(self.variable.dimension, tuple):
                dims = [convert(d) for d in self.variable.dimension]

                if not self.variable.fortran_order:
                    dims = dims[::-1]

                result += dims[0]
                for dim in dims[1:]:
                    result += [",", " "]
                    result += dim
            else:
                result += convert(self.variable.dimension)

            result += [")"]

        if self.variable.intent is not None:
            result += [", ", "intent", "(", self.variable.intent, ")"]

        if settings.force_value:
            if self.variable.dimension is None and self.variable.intent == "in":
                result += [", ", "value"]

        if self.variable.parameter:
            result += [", ", "parameter"]

        assign_str = None

        if self.variable.assign is not None:
            # TODO this is horrible
            from numeta.syntax.expressions import LiteralNode

            if not isinstance(self.variable.assign, list):
                to_assign = self.variable.assign
            else:
                # find the dimensions/shape of assign
                dim_assign = []
                dim_assign.append(len(self.variable.assign))

                if isinstance(self.variable.assign[0], list):
                    dim_assign.append(len(self.variable.assign[0]))

                    if isinstance(self.variable.assign[0][0], list):
                        dim_assign.append(len(self.variable.assign[0][0]))

                        if isinstance(self.variable.assign[0][0][0], list):
                            error_str = "Only assignmets with max rank 3"
                            if self.variable.subroutine is not None:
                                error_str += (
                                    f"\nName of the subroutine: {self.variable.subroutine.name}"
                                )
                            error_str += f"\nName of the self.variable: {self.variable.name}"
                            error_str += (
                                f"\nDimension of the self.variable: {self.variable.dimension}"
                            )
                            error_str += f"\nDimension of the assignment: {tuple(dim_assign[::-1])}"
                            raise Warning(error_str)

                elements_to_assign = []

                if len(dim_assign) == 1:
                    for element_1 in self.variable.assign:
                        if isinstance(element_1, (int, float, complex)):
                            elements_to_assign.append(LiteralNode(element_1))
                        else:
                            elements_to_assign.append(element_1)

                elif len(dim_assign) == 2:
                    for element_1 in self.variable.assign:
                        for element_2 in element_1:
                            if isinstance(element_1, (int, float, complex)):
                                elements_to_assign.append(LiteralNode(element_2))
                            else:
                                elements_to_assign.append(element_2)

                elif len(dim_assign) == 3:
                    for element_1 in self.variable.assign:
                        for element_2 in element_1:
                            for element_3 in element_2:
                                if isinstance(element_1, (int, float, complex)):
                                    elements_to_assign.append(LiteralNode(element_3))
                                else:
                                    elements_to_assign.append(element_3)

                to_assign = ["["]
                for element in elements_to_assign:
                    if hasattr(element, "get_code_blocks"):
                        to_assign += element.get_code_blocks()
                    else:
                        to_assign.append(str(element))
                    to_assign.append(", ")
                to_assign[-1] = "]"

            if self.variable.dimension is None:
                assign_str = [" = ", to_assign]

            elif not isinstance(self.variable.dimension, tuple):
                assign_str = [" = ", *to_assign]

            elif len(self.variable.dimension) == 1:
                assign_str = [" = ", *to_assign]

            else:
                assign_str = [" = ", "reshape", "("]
                assign_str += to_assign
                assign_str.append(", ")
                assign_str.append("[")
                for dim in self.variable.dimension:
                    assign_str += [str(dim), ", "]
                assign_str[-1] = "]"
                assign_str.append(")")

        result += [" :: ", self.variable.name]

        if assign_str is not None:
            result += assign_str

        return result
