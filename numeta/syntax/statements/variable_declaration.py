from .statement import Statement
from numeta.syntax.nodes import NamedEntity
from numeta.syntax.syntax_settings import settings
from .tools import get_shape_blocks


class VariableDeclaration(Statement):
    def __init__(self, variable, add_to_scope=False):
        super().__init__(add_to_scope=add_to_scope)
        self.variable = variable

    def extract_entities(self):
        yield from self.variable.ftype.extract_entities()

        if settings.array_lower_bound != 1:
            # HACK: Non stardard array lower bound so we have to shift it
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
            result += [", ", "dimension"]
            result += ["("] + [":", ","] * (len(self.variable.dimension) - 1) + [":", ")"]
        elif self.variable.pointer:
            result += [", ", "pointer"]
            result += [", ", "dimension"]
            result += ["("] + [":", ","] * (len(self.variable.dimension) - 1) + [":", ")"]
        elif self.variable.dimension is not None:
            result += [", ", "dimension"]
            result += get_shape_blocks(
                self.variable.dimension, fortran_order=self.variable.fortran_order
            )

        if self.variable.intent is not None:
            result += [", ", "intent", "(", self.variable.intent, ")"]

        if settings.force_value:
            if self.variable.dimension is None and self.variable.intent == "in":
                result += [", ", "value"]

        if self.variable.parameter:
            result += [", ", "parameter"]

        if self.variable.target:
            # why fortran? why?
            if self.variable.pointer:
                result += [", ", "contiguous"]
            else:
                result += [", ", "target"]

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
