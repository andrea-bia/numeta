from .nodes import NamedEntity
from .expressions import ExpressionNode


class Variable(NamedEntity, ExpressionNode):
    def __init__(
        self,
        name,
        ftype,
        dimension=None,
        intent=None,
        pointer=False,
        target=False,
        allocatable=False,
        parameter=False,
        assign=None,
        module=None,
        fortran_order=True,
    ):
        super().__init__(name, module=module)
        self.ftype = ftype
        self.dimension = dimension
        if isinstance(dimension, tuple):
            if len(dimension) == 1 and isinstance(dimension[0], (Variable, int)):
                self.dimension = dimension[0]
            else:
                self.dimension = dimension
        self.allocatable = allocatable
        self.parameter = parameter
        self.assign = assign
        self.intent = intent
        self.pointer = pointer
        self.target = target
        self.fortran_order = fortran_order

    @property
    def dtype(self):
        return self.ftype

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
    def T(self):
        from .expressions import Transpose

        return Transpose(self)

    @property
    def shape(self):
        if self.dimension is None:
            raise ValueError("The variable is a scalar")
        return self.dimension

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

    def __getitem__(self, key):
        if isinstance(key, slice) and key.start is None and key.stop is None and key.step is None:
            return self
        from .expressions import GetAttr, GetItem

        if isinstance(key, str):
            return GetAttr(self, key)
        return GetItem(self, key)

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
            self.ftype,
            dimension=self.dimension,
            intent=self.intent,
            pointer=self.pointer,
            target=self.target,
            allocatable=self.allocatable,
            parameter=self.parameter,
            assign=self.assign,
            module=self.module,
            fortran_order=self.fortran_order,
        )
