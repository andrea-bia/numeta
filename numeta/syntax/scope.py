class Scope:
    current_scope = None

    @classmethod
    def add_to_current_scope(cls, statement):
        if cls.current_scope is None:
            raise Exception("No current scope")
        cls.current_scope.__body.append(statement)

    @classmethod
    def end(cls):
        if not issubclass(cls, Scope):
            raise Exception("end() must be called from a subclass of Scope")
        if cls.current_scope is None:
            raise Exception("No current scope")
        cls.current_scope = cls.current_scope.parent

    def __init__(self, parent=None) -> None:
        self.parent = parent if parent is not None else Scope.current_scope
        self.__body = []

    @property
    def body(self):
        return self.__body

    @body.setter
    def body(self, value):
        self.__body = value

    def enter(self):
        Scope.current_scope = self

    def exit(self):
        Scope.current_scope = self.parent

    def get_statements(self):
        return self.body

    def __enter__(self):
        self.enter()

    def __exit__(self, *args):
        self.exit()
