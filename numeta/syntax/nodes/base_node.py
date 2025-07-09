from abc import ABC, abstractmethod


class Node(ABC):
    @abstractmethod
    def get_code_blocks(self):
        pass

    def __str__(self):
        return "".join(self.get_code_blocks())

    @abstractmethod
    def extract_entities(self):
        pass
