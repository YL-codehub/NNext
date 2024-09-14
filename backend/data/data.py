import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split


class Dataset(ABC):
    """Abstract Model class that is inherited to all models"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.fetched = None
        self.X = None
        self.Y = None
        self.X_train_set = None
        self.Y_train_set = None
        self.X_test_set = None
        self.Y_test_set = None

    @abstractmethod
    def fetch(self):
        """Downloads data."""
        pass

    @abstractmethod
    def format(self):
        """Creating X and Y from self.fetched. Normalisations/ feature engineering."""
        pass

    @abstractmethod
    def split_sets(self, **kwargs):
        self.X_train_set, self.X_test_set, self.Y_train_set, self.Y_test_set = train_test_split(self.X, self.Y,**kwargs)

    @abstractmethod
    def store(self):
        pass

    @abstractmethod
    def refresh(self):
        pass
