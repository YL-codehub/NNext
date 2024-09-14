import logging
from abc import ABC, abstractmethod
from model import BaseModel
from backend.data.data_assets import Assets
from utils.torch_tools import MLP

class ModelNNext(BaseModel):
    """Abstract Model class that is inherited to all models"""

    def __init__(self, cfg):
        super().__init__(cfg)

    @abstractmethod
    def load_data(self):
        my_data = Assets(self.cfg.data)
        # X2 = torch.tensor(X_train, dtype=torch.float32)
        # Y2 = torch.tensor(y_train, dtype=torch.float32).reshape(-1, N)  # updated to match y2's new shape
        # X2_test = torch.tensor(X_test, dtype=torch.float32)
        # Y2_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, N)  # updated to match y2's new shape
        pass

    @abstractmethod
    def build(self):
        self.model = MLP(self.cfg.model,)##and what?)
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
