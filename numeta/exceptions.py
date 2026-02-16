class NumetaError(Exception):
    """Base exception for all numeta errors."""

    pass


class CompilationError(NumetaError):
    """Error during compilation."""

    pass


class NumetaTypeError(NumetaError):
    """Type error in numeta code."""

    pass


class NumetaNotImplementedError(NumetaError):
    """Feature not yet implemented."""

    pass


def format_source_location(node):
    """Format source location info from a node for error messages."""
    if node is None:
        return None

    loc = getattr(node, "source_location", None)
    if loc is None:
        return None

    filename = loc.get("filename", "<unknown>")
    lineno = loc.get("lineno", 0)

    # Try to get the actual line of code
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
            if 0 <= lineno - 1 < len(lines):
                source_line = lines[lineno - 1].rstrip()
                return f'  File "{filename}", line {lineno}\n    {source_line}'
    except Exception:
        pass

    return f'  File "{filename}", line {lineno}'


def raise_with_source(exception_class, message, source_node=None):
    """Raise an exception with source location information.

    Args:
        exception_class: The exception class to raise (e.g., NotImplementedError)
        message: The error message
        source_node: The AST/IR node that caused the error (should have source_location)
    """
    loc_info = format_source_location(source_node)
    if loc_info:
        full_message = f"{message}\n\n{loc_info}"
    else:
        full_message = message

    raise exception_class(full_message)
