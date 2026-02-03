from .statement import StatementWithScope
from .variable_declaration import VariableDeclaration
from numeta.ast.variable import Variable


class DerivedTypeDeclaration(StatementWithScope):
    def __init__(self, derived_type):
        super().__init__(enter_scope=False, add_to_scope=False)
        self.derived_type = derived_type

    @property
    def children(self):
        return []

    def get_statements(self):
        for name, fortran_type, shape in self.derived_type.fields:
            yield VariableDeclaration(Variable(name, ftype=fortran_type, shape=shape))
