import numpy as np
import numbers


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
    if isinstance(node, (numbers.Number, np.generic, bool, str)):
        from .expressions import LiteralNode

        return LiteralNode(node)
    else:
        return node
        # otherwise is so slow
        # TODO: maybe to move check node at the print time
        from .nodes import Node

        if isinstance(node, Node):
            return node
        else:
            raise ValueError(f"Unknown node type: {node.__class__.__name__} value: {node}")


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
            tuple(update_variables(dim, variables_couples) for dim in element.dims),
            fortran_order=element.fortran_order,
        )
    from numeta.syntax.nodes.base_node import Node

    if isinstance(element, Node):
        return element.get_with_updated_variables(variables_couples)
    return element
