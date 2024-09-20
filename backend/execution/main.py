from backend.models.model_nnext import ModelNNext
from configs.config import get_cfg
import matplotlib.pyplot as plt

cfg = get_cfg("model_finance.yaml")

# Instantiate and print the model
model = ModelNNext(cfg)
model.run_sequence()

Ypred = model.evaluate(model.data.X_test_set.reshaped_2D_tensor()).numpy()
Ytrue = model.data.Y_test_set.ndarray
floorYtest = model.data.Ybis_test_set.ndarray

print(Ypred)
Ypred= (Ypred/100+1)*floorYtest
Ytrue= (Ytrue/100+1)*floorYtest

plt.plot(Ypred)
plt.plot(Ytrue, '--')

plt.show()