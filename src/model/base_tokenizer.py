"""Abstract Tokenizer"""

from abc import ABC, abstractmethod

class BaseTokenizer(ABC):
    """Abstract tokenizer class that is inherited to all tokenizers"""
    def __init__(self, cfg):
        self.config = cfg

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def tokenize(self, string):
        pass
