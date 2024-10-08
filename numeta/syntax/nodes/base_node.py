from abc import ABC, abstractmethod


class Node(ABC):
    @abstractmethod
    def get_code_blocks(self):
        pass

    @abstractmethod
    def extract_entities(self):
        pass
