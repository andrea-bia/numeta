from .statement import StatementWithScope
from .variable_declaration import VariableDeclaration
from numeta.syntax.variable import Variable
from numeta.syntax.syntax_settings import settings


class DerivedTypeDeclaration(StatementWithScope):
    def __init__(self, derived_type):
        super().__init__(enter_scope=False, add_to_scope=False)
        self.derived_type = derived_type

    @property
    def children(self):
        return []

    def get_statements(self):
        for name, fortran_type, dimension in self.derived_type.fields:
            yield VariableDeclaration(Variable(name, ftype=fortran_type, dimension=dimension))

    def get_start_code_blocks(self):
        if settings.derived_type_bind_c:
            return ["type", ", ", "bind(C)", " ", "::", " ", self.derived_type.name]
        return ["type", " ", "::", " ", self.derived_type.name]

    def get_end_code_blocks(self):
        return ["end", " ", "type", " ", self.derived_type.name]
