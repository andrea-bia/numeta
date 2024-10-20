from numeta.syntax.tools import check_node
from numeta.syntax.scope import Scope
from numeta.syntax.syntax_settings import settings
from .tools import print_block, get_shape_blocks
from .statement import Statement, StatementWithScope


class Comment(Statement):
    def __init__(self, comment, add_to_scope=False):
        super().__init__(add_to_scope=add_to_scope)
        self.comment = comment
        if isinstance(comment, str):
            self.comment = [comment]

    @property
    def children(self):
        return []

    def print_lines(self, indent=0):
        """Print the statement, formatted with the given indent level."""
        return [print_block(self.comment, indent=indent, prefix="! ")]


class Use(Statement):
    def __init__(self, module, only=None, add_to_scope=True):
        super().__init__(add_to_scope=add_to_scope)
        self.module = module
        self.only = only

    @property
    def children(self):
        return []

    def get_code_blocks(self):
        result = ["use", " ", self.module.name]
        if self.only is not None:
            result += [", ", "only", ": ", self.only.name]
        return result


class Implicit(Statement):
    def __init__(self, implicit_type="none", add_to_scope=True):
        super().__init__(add_to_scope=add_to_scope)
        self.implicit_type = implicit_type

    @property
    def children(self):
        return []

    def get_code_blocks(self):
        return ["implicit", " ", self.implicit_type]


class Assignment(Statement):
    def __init__(self, assignment_target, value, add_to_scope=True):
        super().__init__(add_to_scope=add_to_scope)
        self.target = check_node(assignment_target)
        self.value = check_node(value)

    @property
    def children(self):
        return [self.target, self.value]

    def get_code_blocks(self):
        result = self.target.get_code_blocks()
        result.append("=")
        result += self.value.get_code_blocks()
        return result


class SimpleStatement(Statement):
    token = ""

    def __init__(self):
        Scope.add_to_current_scope(self)

    @property
    def children(self):
        return []

    def get_code_blocks(self):
        return [self.__class__.token]


class Cycle(SimpleStatement):
    token = "cycle"


class Exit(SimpleStatement):
    token = "exit"


class Stop(SimpleStatement):
    token = "stop"


class Return(SimpleStatement):
    token = "return"


class Print(Statement):
    def __init__(self, *children):
        super().__init__()
        self.to_print = [check_node(child) for child in children]

    @property
    def children(self):
        return self.to_print

    def get_code_blocks(self):
        result = ["print *, "]
        for child in self.children:
            if isinstance(child, str):
                protected_child = child.replace('"', '""')
                result.append(f'"{protected_child}"')
            else:
                result += child.get_code_blocks()
            result.append(", ")
        result.pop()
        return result


class Allocate(Statement):
    def __init__(self, target, *shape):
        super().__init__()
        self.target = check_node(target)
        self.shape = [check_node(child) for child in shape]

    @property
    def children(self):
        return [self.target] + self.shape

    def get_code_blocks(self):
        result = ["allocate", "("]

        result += self.target.get_code_blocks()

        dims = []
        for argument in self.shape:
            if (lbound := settings.array_lower_bound) != 1:
                dims.append([str(lbound), ":", *(argument + (lbound - 1)).get_code_blocks()])
            else:
                dims.append([*(argument).get_code_blocks()])

        if not self.target.fortran_order:
            dims = dims[::-1]

        result.append("(")
        result += dims[0]
        for dim in dims[1:]:
            result += [",", " "]
            result += dim
        result.append(")")

        result.append(")")

        return result

        # contains some lists
        # new_result = []
        # for element in result:
        #    if type(element) is list:
        #        new_result += element
        #    else:
        #        new_result.append(element)

        # return new_result


class Deallocate(Statement):
    def __init__(self, array):
        super().__init__()
        self.array = check_node(array)

    @property
    def children(self):
        return [self.array]

    def get_code_blocks(self):
        result = ["deallocate", "(", *self.array.get_code_blocks(), ")"]
        return result


class Do(StatementWithScope):
    def __init__(self, iterator, start, end, step=None):
        super().__init__()
        self.iterator = check_node(iterator)
        self.start = check_node(start)
        self.end = check_node(end)
        self.step = None if step is None else check_node(step)

    @property
    def children(self):
        return [self.iterator, self.start, self.end] + (
            [self.step] if self.step is not None else []
        )

    def get_start_code_blocks(self):
        result = ["do", " "]
        result += self.iterator.get_code_blocks()
        result += [" ", "=", " "]
        result += self.start.get_code_blocks()
        result.append(", ")
        result += self.end.get_code_blocks()
        if len(self.children) == 4:
            result.append(", ")
            result += self.step.get_code_blocks()

        return result

    def get_end_code_blocks(self):
        return ["end", " ", "do"]


class DoWhile(StatementWithScope):
    def __init__(self, condition):
        super().__init__()
        self.condition = check_node(condition)

    @property
    def children(self):
        return [self.condition]

    def get_start_code_blocks(self):
        return ["do while", " ", "(", *self.condition.get_code_blocks(), ")"]

    def get_end_code_blocks(self):
        return ["end", " ", "do"]


class If(StatementWithScope):
    def __init__(self, condition):
        super().__init__()
        self.condition = check_node(condition)
        self.orelse = []

    @property
    def children(self):
        return [self.condition]

    def get_statements(self):
        return self.scope.get_statements() + self.orelse

    def print_lines(self, indent=0):
        """Print the entire scoped statement, including all nested statements."""
        result = [print_block(self.get_start_code_blocks(), indent=indent)]
        for statement in self.scope.get_statements():
            result.extend(statement.print_lines(indent=indent + 1))
        for statement in self.orelse:
            result.extend(statement.print_lines(indent=indent))
        result.append(print_block(self.get_end_code_blocks(), indent=indent))
        return result

    def get_start_code_blocks(self):
        return ["if", "(", *self.condition.get_code_blocks(), ")", "then"]

    def get_end_code_blocks(self):
        return ["end", " ", "if"]


class ElseIf(StatementWithScope):
    def __init__(self, condition):
        if not isinstance(Scope.current_scope.body[-1], If):
            raise Exception(
                "Something went wrong with this else if. The last statement is not an if statement."
            )
        Scope.current_scope.body[-1].orelse.append(self)
        self.scope = Scope()
        self.scope.enter()
        self.condition = check_node(condition)

    @property
    def children(self):
        return [self.condition]

    def get_start_code_blocks(self):
        return ["elseif", "(", *self.condition.get_code_blocks(), ")", "then"]

    def get_end_code_blocks(self):
        return []


class Else(StatementWithScope):
    def __init__(self):
        if not isinstance(Scope.current_scope.body[-1], If):
            raise Exception(
                "Something went wrong with this else if. The last statement is not an if statement."
            )
        Scope.current_scope.body[-1].orelse.append(self)
        self.scope = Scope()
        self.scope.enter()

    @property
    def children(self):
        return []

    def get_start_code_blocks(self):
        return ["else"]

    def get_end_code_blocks(self):
        return []


class SelectCase(StatementWithScope):
    def __init__(self, value):
        super().__init__()
        self.value = check_node(value)

    @property
    def children(self):
        return [self.value]

    def get_start_code_blocks(self):
        return ["select", " ", "case", " ", "(", *self.value.get_code_blocks(), ")"]

    def get_end_code_blocks(self):
        return ["end", " ", "select"]


class Case(StatementWithScope):
    def __init__(self, value):
        super().__init__()
        self.value = check_node(value)

    @property
    def children(self):
        return [self.value]

    def get_start_code_blocks(self):
        result = ["case", " ", "("]
        result += self.value.get_code_blocks()
        result.append(")")
        return result

    def get_end_code_blocks(self):
        return []


class Contains(Statement):
    def __init__(self, add_to_scope=True):
        super().__init__(add_to_scope=add_to_scope)

    @property
    def children(self):
        return []

    def get_code_blocks(self):
        return ["contains"]


class Interface(StatementWithScope):
    def __init__(self, methods):
        super().__init__(add_to_scope=False)
        self.methods = methods

    @property
    def children(self):
        return []

    def get_statements(self):
        result = []
        for method in self.methods:
            result.append(method.get_interface_declaration())
        return result

    def get_start_code_blocks(self):
        return ["interface"]

    def get_end_code_blocks(self):
        return ["end", " ", "interface"]


class PointerAssignment(Statement):
    def __init__(self, pointer, pointer_shape, target, target_shape=None):
        super().__init__()
        self.pointer = check_node(pointer)
        self.pointer_shape = []
        # should specify bounds for the pointer
        for dim in pointer_shape:
            if not isinstance(dim, slice):
                self.pointer_shape.append(slice(None, dim))
            else:
                self.pointer_shape.append(dim)
        self.pointer_shape = tuple(self.pointer_shape)
        self.target = check_node(target)
        self.target_shape = target_shape

        from numeta.syntax.variable import Variable

        if not isinstance(self.target, Variable):
            raise Exception("The target of a pointer must be a variable.")
        self.target.target = True

        if not isinstance(self.pointer, Variable):
            raise Exception("The pointer must be a variable.")
        self.pointer.pointer = True

    @property
    def children(self):
        return [self.target, self.pointer]

    def get_code_blocks(self):
        return [
            *self.pointer.get_code_blocks(),
            *get_shape_blocks(self.pointer_shape),
            "=>",
            *self.target.get_code_blocks(),
        ]
