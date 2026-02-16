from numeta.ast.tools import check_node
from numeta.ast.scope import Scope
from numeta.exceptions import raise_with_source
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


class Import(Statement):
    def __init__(self, module, only=None, add_to_scope=True):
        super().__init__(add_to_scope=add_to_scope)
        self.module = module
        self.only = only

    @property
    def children(self):
        return []


class TypingPolicy(Statement):
    def __init__(self, implicit_type="none", add_to_scope=True):
        super().__init__(add_to_scope=add_to_scope)
        self.implicit_type = implicit_type

    @property
    def children(self):
        return []


class Assignment(Statement):
    def __init__(self, assignment_target, value, add_to_scope=True):
        super().__init__(add_to_scope=add_to_scope)
        self.target = check_node(assignment_target)
        self.value = check_node(value)

    @property
    def children(self):
        return [self.target, self.value]


class SimpleStatement(Statement):
    token = ""

    def __init__(self):
        super().__init__(add_to_scope=True)

    @property
    def children(self):
        return []


class Continue(SimpleStatement):
    token = "continue"


class Break(SimpleStatement):
    token = "break"


class Halt(SimpleStatement):
    token = "stop"


class Return(SimpleStatement):
    token = "return"


class Print(Statement):
    def __init__(self, *children, add_to_scope=True):
        super().__init__(add_to_scope=add_to_scope)
        self.to_print = [check_node(child) for child in children]

    @property
    def children(self):
        return self.to_print


class Allocate(Statement):
    def __init__(self, target, *shape, add_to_scope=True):
        super().__init__(add_to_scope=add_to_scope)
        self.target = check_node(target)
        self.shape = [check_node(child) for child in shape]

    @property
    def children(self):
        return [self.target] + self.shape


class Deallocate(Statement):
    def __init__(self, array, add_to_scope=True):
        super().__init__(add_to_scope=add_to_scope)
        self.array = check_node(array)

    @property
    def children(self):
        return [self.array]


class For(StatementWithScope):
    def __init__(self, iterator, start, end, /, step=None, *, add_to_scope=True, enter_scope=True):
        super().__init__(add_to_scope=add_to_scope, enter_scope=enter_scope)
        self.iterator = check_node(iterator)
        self.start = check_node(start)
        self.end = check_node(end)
        self.step = None if step is None else check_node(step)

    @property
    def children(self):
        return [self.iterator, self.start, self.end] + (
            [self.step] if self.step is not None else []
        )


class While(StatementWithScope):
    def __init__(self, condition, /, *, add_to_scope=True, enter_scope=True):
        super().__init__(add_to_scope=add_to_scope, enter_scope=enter_scope)
        self.condition = check_node(condition)

    @property
    def children(self):
        return [self.condition]


class If(StatementWithScope):
    def __init__(self, condition, /, *, add_to_scope=True, enter_scope=True):
        super().__init__(add_to_scope=add_to_scope, enter_scope=enter_scope)
        self.condition = check_node(condition)
        self.orelse = []

    @property
    def children(self):
        return [self.condition]

    def get_statements(self):
        return self.scope.get_statements() + self.orelse

    def get_with_updated_variables(self, variables_couples):
        new_children = [
            child.get_with_updated_variables(variables_couples) for child in self.children
        ]
        result = type(self)(*new_children, add_to_scope=False, enter_scope=False)
        result.scope = self.scope.get_with_updated_variables(variables_couples)
        result.orelse = [stmt.get_with_updated_variables(variables_couples) for stmt in self.orelse]
        return result


class ElseIf(StatementWithScope):
    def __init__(self, condition, /, *, add_to_scope=True, enter_scope=True):
        super().__init__(add_to_scope=False, enter_scope=False)
        if add_to_scope:
            if not isinstance(Scope.current_scope.body[-1], If):
                raise_with_source(
                    Exception,
                    "Something went wrong with this else if. The last statement is not an if statement.",
                    source_node=self,
                )
            Scope.current_scope.body[-1].orelse.append(self)
        if enter_scope:
            self.scope.enter()
        self.condition = check_node(condition)

    @property
    def children(self):
        return [self.condition]


class Else(StatementWithScope):
    def __init__(self, /, *, add_to_scope=True, enter_scope=True):
        super().__init__(add_to_scope=False, enter_scope=False)
        if add_to_scope:
            if not isinstance(Scope.current_scope.body[-1], If):
                raise_with_source(
                    Exception,
                    "Something went wrong with this else if. The last statement is not an if statement.",
                    source_node=self,
                )
            Scope.current_scope.body[-1].orelse.append(self)
        if enter_scope:
            self.scope.enter()

    @property
    def children(self):
        return []


class Switch(StatementWithScope):
    def __init__(self, value, add_to_scope=True):
        super().__init__(add_to_scope=add_to_scope)
        self.value = check_node(value)

    @property
    def children(self):
        return [self.value]


class Case(StatementWithScope):
    def __init__(self, value, add_to_scope=True):
        super().__init__(add_to_scope=add_to_scope)
        self.value = check_node(value)

    @property
    def children(self):
        return [self.value]


class Section(Statement):
    def __init__(self, add_to_scope=True):
        super().__init__(add_to_scope=add_to_scope)

    @property
    def children(self):
        return []


class InterfaceBlock(StatementWithScope):
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


class PointerAssignment(Statement):
    def __init__(self, pointer, pointer_shape, target, target_shape=None, add_to_scope=True):
        super().__init__(add_to_scope=add_to_scope)
        self.pointer = check_node(pointer)
        self.pointer_shape = []
        # should specify bounds for the pointer
        for dim in pointer_shape.dims:
            if not isinstance(dim, slice):
                self.pointer_shape.append(slice(None, dim))
            else:
                self.pointer_shape.append(dim)
        if not pointer_shape.fortran_order:
            self.pointer_shape.reverse()
        self.pointer_shape = tuple(self.pointer_shape)
        self.target = check_node(target)
        self.target_shape = target_shape

        from numeta.ast.variable import Variable
        from numeta.ast.expressions import GetItem

        if isinstance(self.target, Variable):
            self.target.target = True
        elif isinstance(self.target, GetItem):
            self.target.variable.target = True
        else:
            raise_with_source(
                Exception,
                "The target of a pointer must be a variable or GetItem.",
                source_node=self,
            )

        if not isinstance(self.pointer, Variable):
            raise_with_source(
                Exception,
                "The pointer must be a variable.",
                source_node=self,
            )
        self.pointer.pointer = True

    @property
    def children(self):
        return [self.target, self.pointer]

    def get_with_updated_variables(self, variables_couples):
        def update_variables(element):
            if isinstance(element, tuple):
                return tuple(update_variables(e) for e in element)
            if isinstance(element, slice):
                return slice(
                    update_variables(element.start),
                    update_variables(element.stop),
                    update_variables(element.step),
                )
            from numeta.array_shape import ArrayShape

            if isinstance(element, ArrayShape):
                return ArrayShape(
                    tuple(update_variables(dim) for dim in element.dims),
                    fortran_order=element.fortran_order,
                )
            from numeta.ast.nodes.base_node import Node

            if isinstance(element, Node):
                return element.get_with_updated_variables(variables_couples)
            return element

        new_pointer = self.pointer.get_with_updated_variables(variables_couples)
        new_target = self.target.get_with_updated_variables(variables_couples)
        from numeta.array_shape import ArrayShape

        new_pointer_shape = ArrayShape(update_variables(self.pointer_shape), fortran_order=True)
        new_target_shape = (
            update_variables(self.target_shape) if self.target_shape is not None else None
        )
        return type(self)(
            new_pointer,
            new_pointer_shape,
            new_target,
            new_target_shape,
            add_to_scope=False,
        )
