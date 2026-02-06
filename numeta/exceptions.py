class NumetaError(Exception):
    pass


class CompilationError(NumetaError):
    pass


class NumetaTypeError(NumetaError):
    pass


class NumetaNotImplementedError(NumetaError):
    pass
