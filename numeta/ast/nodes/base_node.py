from abc import ABC, abstractmethod


class Node(ABC):

    @abstractmethod
    def extract_entities(self):
        """Extract the nested entities of the node."""
