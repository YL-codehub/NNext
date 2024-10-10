import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from utils.data_tools import GenericArray


class Dataset(ABC):
    """Abstract Model class that is inherited to all models"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.fetched = None

        # time x events
        self.X = GenericArray()
        self.Y = GenericArray()
        self.X_train_set = GenericArray()
        self.Y_train_set = GenericArray()
        self.X_test_set = GenericArray()
        self.Y_test_set = GenericArray()

        ## buffer arrays, eg to store og norm or floor value...
        self.Xbis = GenericArray()
        self.Ybis = GenericArray()
        self.Xbis_train_set = GenericArray()
        self.Ybis_train_set = GenericArray()
        self.Xbis_test_set = GenericArray()
        self.Ybis_test_set = GenericArray()
        self.scal_buffer_train = None
        self.scal_buffer_test = None

    @abstractmethod
    def fetch(self):
        """Downloads data."""
        pass

    @abstractmethod
    def format(self):
        """Creating X and Y from self.fetched. Normalisations/ feature engineering."""
        pass

    @abstractmethod
    def store(self):
        pass

    @abstractmethod
    def refresh(self):
        pass

    def split_sets(self, **kwargs):
        kwargs.setdefault('test_size', .25)
        kwargs.setdefault('shuffle', False) ## why not shuffling? to do in training batches at least.
        self.X_train_set.ndarray, self.X_test_set.ndarray, self.Y_train_set.ndarray, self.Y_test_set.ndarray = train_test_split(self.X.ndarray, self.Y.ndarray,**kwargs)
        self.Xbis_train_set.ndarray, self.Xbis_test_set.ndarray, self.Ybis_train_set.ndarray, self.Ybis_test_set.ndarray = train_test_split(self.Xbis.ndarray, self.Ybis.ndarray,**kwargs)
        self.scal_buffer_train = np.max(self.Y_train_set.ndarray,axis=0)
        self.scal_buffer_test = np.max(self.Y_test_set.ndarray,axis = 0)
        self.X_train_set.ndarray /= self.scal_buffer_train
        self.X_test_set.ndarray /= self.scal_buffer_test
        self.Y_train_set.ndarray /= self.scal_buffer_train
        self.Y_test_set.ndarray /= self.scal_buffer_test
        
    def sequence(self, **kwargs):
        self.fetch()
        self.format()
        self.split_sets(**kwargs)

