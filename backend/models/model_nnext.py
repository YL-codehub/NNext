import logging
from abc import ABC, abstractmethod
from backend.models.model import BaseModel
from backend.data.dataset_assets import Assets
from utils.torch_tools import MLP

class ModelNNext(BaseModel):
    """Abstract Model class that is inherited to all models"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.data = Assets(self.cfg.data)
        self.X_train = None
        self.Y_train = None

    def build(self):
        self.X_train = self.data.X_train_set.reshaped_2D_tensor()
        self.Y_train = self.data.Y_train_set.reshaped_tensor((-1,self.data.X_train_set.ndarray.shape[2]))
        self.model = MLP(self.cfg.model,(self.X_train.shape[1],self.data.Y_train_set.ndarray.shape[1]))

    def train(self):
        self.model.train(self.X_train,self.Y_train)

    def evaluate(self, X):
        return self.model.predict(X)


if __name__ == "__main__":
    from configs.config import get_cfg
    cfg = get_cfg("model_finance.yaml")
    # Instantiate and print the model
    model = ModelNNext(cfg)
    model.prepare_data()
    model.build()
    print(model.model)