from abc import ABC, abstractmethod
from .base_node import Node


class NamedEntity(Node, ABC):
    """
    A named entity that can be referenced.

    Attributes
    ----------
    name : str
        The name of the entity.
    parent : None | Module | ExternalLibrary
        If the entity is local the parent is None, if it is a module variable it is a module, lastly if it ca be a library.

    Methods
    -------
    extract_entities():
        Extract the entity itself.
    get_code_blocks():
        Return the code blocks representing the entity.
    """

    def __init__(self, name, parent=None) -> None:
        self.name = name
        self.parent = parent

    def __hash__(self):
        return hash(self.name)

    @abstractmethod
    def get_declaration(self):
        pass

    def get_code_blocks(self):
        return [self.name]

    def extract_entities(self):
        yield self
