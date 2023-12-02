"""Abstract base model"""

from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract Model class that is inherited to all models"""
    def __init__(self, cfg):
        self.config = cfg

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def save_weights(self):
        pass

    @abstractmethod
    def load_weights(self):
        pass

    @abstractmethod
    def train(self):
        pass