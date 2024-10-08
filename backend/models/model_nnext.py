import logging
from abc import ABC, abstractmethod
from backend.models.model import BaseModel
from backend.data.dataset_assets import Assets
# from utils.torch_tools import MLP
from utils.torch_tools import GeneralNN


class ModelNNext(BaseModel):
    """Abstract Model class that is inherited to all models"""

    def __init__(self, cfg, flat_input = True):
        super().__init__(cfg)
        self.data = Assets(self.cfg.data)
        self.X_train = None
        self.Y_train = None
        self.X_valid = None
        self.Y_valid = None
        self.flat_input = flat_input

    def build(self):
        if self.flat_input:
            self.X_train = self.data.X_train_set.reshaped_2D_tensor()
            self.X_valid = self.data.X_test_set.reshaped_2D_tensor()
        else:
            self.X_train = self.data.X_train_set.tensorT #.unsqueeze(1)
            self.X_valid = self.data.X_test_set.tensorT #.unsqueeze(1)
        self.Y_train = self.data.Y_train_set.tensor # reshaped_tensor((-1,self.data.X_train_set.ndarray.shape[2]))
        self.Y_valid = self.data.Y_test_set.tensor
        # self.model = MLP(self.cfg.model,(self.X_train.shape[1],self.data.Y_train_set.ndarray.shape[1]))
        self.model = GeneralNN(self.cfg.model)


    def train(self, verbose = True):
        loss_valid = self.model.train(self.X_train,self.Y_train,self.X_valid, self.Y_valid, verbose)
        return loss_valid

    def evaluate(self, X):
        return self.model.predict(X)


if __name__ == "__main__":
    from configs.config import get_cfg
    cfg = get_cfg("model_finance.yaml")
    # Instantiate and print the model
    model = ModelNNext(cfg)
    model.prepare_data()
    model.build()
    print(model.evaluate(model.data.X_test_set.reshaped_2D_tensor()))