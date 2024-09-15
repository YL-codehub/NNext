import logging
from abc import ABC, abstractmethod
from backend.data.dataset import Dataset


class BaseModel(ABC):
    """Abstract Model class that is inherited to all models"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.data = None
        self.model = None

    def prepare_data(self):
        self.data.sequence()

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
