from abc import ABC, abstractmethod
from .base_node import Node


class NamedEntity(Node, ABC):
    """
    A named entity that can be referenced.

    Attributes
    ----------
    name : str
        The name of the entity.
    parent : None | Namespace | ExternalLibrary
        If the entity is local the parent is None, if it is a namespace variable it is a namespace, lastly if it can be a library.

    Methods
    -------
    extract_entities():
        Extract the entity itself.
    """

    def __init__(self, name, parent=None) -> None:
        self.name = name
        self.parent = parent

    def __hash__(self):
        return hash(self.name)

    @abstractmethod
    def get_declaration(self):
        pass

    def extract_entities(self):
        yield self
