from .syntax import Variable, Scope


class BuilderHelper:
    current_builder = None

    @classmethod
    def set_current_builder(cls, builder):
        cls.current_builder = builder

    @classmethod
    def get_current_builder(cls):
        if cls.current_builder is None:
            raise Warning("The current builder is not initialized")
        return cls.current_builder

    def __init__(self, subroutine, construct_function):
        self.subroutine = subroutine
        self.construct = construct_function
        self.prefix_counter = {}

        # Dictionary to store temporary variables during construction
        # self.tmp = {}

    def generate_local_variables(self, prefix, **kwargs):
        if prefix not in self.prefix_counter:
            self.prefix_counter[prefix] = 0
        self.prefix_counter[prefix] += 1
        return Variable(f"{prefix}{self.prefix_counter[prefix]}", **kwargs)

    def build(self, *args):
        old_builder = self.current_builder
        self.set_current_builder(self)

        old_scope = Scope.current_scope
        self.subroutine.scope.enter()

        self.construct(*args)

        self.subroutine.scope.exit()
        Scope.current_scope = old_scope

        self.set_current_builder(old_builder)
