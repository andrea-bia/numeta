from abc import ABC, abstractmethod


class Node(ABC):
    @abstractmethod
    def get_code_blocks(self):
        """Return the code blocks representing the node."""
        raise NotImplementedError(
            f"Subclass '{self.__class__.__name__}' must implement 'get_code_blocks'."
        )

    def __str__(self):
        return "".join(self.get_code_blocks())

    @abstractmethod
    def extract_entities(self):
        """Extract the nested entities of the node."""
