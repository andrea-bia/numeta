import numpy as np


_SCALAR_LITERAL_TYPES = (bool, int, float, complex, str)
_LiteralNode = None


def _get_literal_node_class():
    global _LiteralNode
    if _LiteralNode is None:
        from .expressions import LiteralNode

        _LiteralNode = LiteralNode
    return _LiteralNode


def extract_entities(element):
    """Yield entities referenced by ``element``, recursively walking containers and slices."""
    if hasattr(element, "extract_entities"):
        yield from element.extract_entities()
    elif isinstance(element, (tuple, list)):
        for e in element:
            yield from extract_entities(e)
    elif isinstance(element, slice):
        yield from extract_entities(element.start)
        yield from extract_entities(element.stop)
        yield from extract_entities(element.step)


def check_node(node):
    """Return a LiteralNode for scalars, otherwise pass through existing nodes."""
    node_type = type(node)
    if node_type in _SCALAR_LITERAL_TYPES or isinstance(node, np.generic):
        return _get_literal_node_class()(node)
    else:
        return node


def update_variables(element, variables_couples):
    """Recursively replace variables in ``element`` using ``variables_couples``."""
    if isinstance(element, tuple):
        return tuple(update_variables(e, variables_couples) for e in element)
    if isinstance(element, list):
        return [update_variables(e, variables_couples) for e in element]
    if isinstance(element, slice):
        return slice(
            update_variables(element.start, variables_couples),
            update_variables(element.stop, variables_couples),
            update_variables(element.step, variables_couples),
        )
    from numeta.array_shape import ArrayShape

    if isinstance(element, ArrayShape):
        return ArrayShape(
            tuple(update_variables(dim, variables_couples) for dim in element.iter_dims()),
            fortran_order=element.fortran_order,
        )
    from numeta.ast.nodes.base_node import Node

    if isinstance(element, Node):
        return element.get_with_updated_variables(variables_couples)
    return element
