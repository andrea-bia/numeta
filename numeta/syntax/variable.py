from .nodes import NamedEntity
from .expressions import ExpressionNode
from numeta.array_shape import ArrayShape, SCALAR


class Variable(NamedEntity, ExpressionNode):
    def __init__(
        self,
        name,
        ftype,
        shape=SCALAR,
        intent=None,
        pointer=False,
        target=False,
        allocatable=False,
        parameter=False,
        assign=None,
        parent=None,
        bind_c=False,
    ):
        super().__init__(name, parent=parent)
        self.__ftype = ftype
        if not isinstance(shape, ArrayShape):
            self.__shape = ArrayShape(shape, fortran_order=True)
        else:
            self.__shape = shape
        self.allocatable = allocatable
        self.parameter = parameter
        self.assign = assign
        self.intent = intent
        self.pointer = pointer
        self.target = target
        # Note that bind c make the variable global
        self.bind_c = bind_c

        from .module import Module

        if isinstance(self.parent, Module):
            self.parent.add_variable(self)

    @property
    def _ftype(self):
        return self.__ftype

    @property
    def _shape(self):
        return self.__shape

    def _set_shape(self, shape):
        if not isinstance(shape, ArrayShape):
            self.__shape = ArrayShape(shape, fortran_order=self._shape.fortran_order)
        else:
            self.__shape = shape

    def get_with_updated_variables(self, variables_couples):
        for old_variable, new_variable in variables_couples:
            if old_variable.name == self.name:
                return new_variable
        return self

    def get_declaration(self):
        from .statements import VariableDeclaration

        return VariableDeclaration(self)

    @property
    def real(self):
        from .expressions import Re

        return Re(self)

    @real.setter
    def real(self, value):
        from .expressions import Re
        from .statements import Assignment

        return Assignment(Re(self), value)

    @property
    def imag(self):
        from .expressions import Im

        return Im(self)

    @imag.setter
    def imag(self, value):
        from .expressions import Im
        from .statements import Assignment

        return Assignment(Im(self), value)

    @property
    def shape(self):
        return self._shape.dims

    def __setitem__(self, key, value):
        """Does nothing, but allows to use variable[key] = value"""
        from .statements import Assignment

        if isinstance(key, slice) and key.start is None and key.stop is None and key.step is None:
            # if the variable is assigned to itself, do nothing, needed for the += and -= operators
            if self is value:
                return
            Assignment(self, value)
        else:
            Assignment(self[key], value)

    def __ilshift__(self, other):
        from .statements import Assignment

        Assignment(self, other)
        return self

    def __iadd__(self, other):
        from .statements import Assignment

        Assignment(self, self + other)
        # need to return same, no real assignment
        return self

    def __isub__(self, other):
        from .statements import Assignment

        Assignment(self, self - other)
        # need to return same, no real assignment
        return self

    def copy(self):
        return Variable(
            self.name,
            self._ftype,
            shape=self._shape,
            intent=self.intent,
            pointer=self.pointer,
            target=self.target,
            allocatable=self.allocatable,
            parameter=self.parameter,
            assign=self.assign,
            parent=self.parent,
        )
