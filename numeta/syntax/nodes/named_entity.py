from abc import ABC, abstractmethod
from .base_node import Node


class NamedEntity(Node, ABC):
    """
    A named entity that can be referenced.

    Attributes
    ----------
    name : str
        The name of the entity.
    module : Module
        The module containing the entity.

    Methods
    -------
    extract_entities():
        Extract the entity.
    get_code_blocks():
        Return the code blocks representing the entity.
    """

    def __init__(self, name, module=None) -> None:
        self.name = name
        self.module = module

    def __hash__(self):
        return hash(self.name)

    @abstractmethod
    def get_declaration(self):
        pass

    def get_code_blocks(self):
        return [self.name]

    def extract_entities(self):
        yield self
