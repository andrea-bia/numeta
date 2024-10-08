def extract_entities(element):
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
    if isinstance(node, (int, float, complex, bool, str)):
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
