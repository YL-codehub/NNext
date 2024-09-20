import torch
import torch.nn as nn
import torch.optim as optim

# Extract model type and layer configurations


def get_model(model):
    models = {
        'Sequential': nn.Sequential,
        # Add more activation functions as needed
    }
    return models.get(model, nn.Sequential)  # Default to Identity if not found


def get_activation(activation_name):
    activations = {
        'ReLU': nn.ReLU,
        'Sigmoid': nn.Sigmoid,
        # Add more activation functions as needed
    }
    return activations.get(activation_name, nn.Identity)  # Default to Identity if not found

def get_layer(layer):
    layers = {
        'Linear': nn.Linear,
        # Add more activation functions as needed
    }
    return layers.get(layer, nn.Linear)  # Default to Identity if not found

def get_loss_func(loss):
    functions = {
        'MSE': nn.MSELoss(),
        # Add more activation functions as needed
    }
    return functions.get(loss, nn.MSELoss())  # Default to Identity if not found


# Create the model class
class MLP:
    def __init__(self, archi, external_size):
        self.layers = nn.ModuleList()
        self.core = archi.layers.hidden_layers

        self.input_size = external_size[0] # updated input size
        self.output_size = external_size[1]  # updated output size

        self.layers.append(get_layer(archi.layers.input[0])(self.input_size, self.core.per_layer*self.input_size))  # updated input dimension
        self.layers.append(get_activation(archi.layers.input[1])())
        for i in range(self.core.count):
            self.layers.append(get_layer(self.core.type[0])(self.core.per_layer*self.input_size, self.core.per_layer*self.input_size))
            self.layers.append(get_activation(self.core.type[1])())
        self.layers.append(get_layer(archi.layers.output[0])(self.core.per_layer*self.input_size, self.output_size))  # updated output dimension
        if len(archi.layers.output)==2:
            self.layers.append(get_activation(archi.layers.output[1])())
        self.model = get_model(archi.type)(*self.layers)

        self.loss_fn   = get_loss_func(archi.training.loss)
        self.n_epochs   = archi.training.n_epochs
        self.batch_size   = archi.training.batch_size

        self.verbose = 0

    def train(self,Xtrain,Ytrain):
        # train the model
         # binary cross entropy
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(self.n_epochs):
            for i in range(0, len(Xtrain), self.batch_size):
                Xbatch = Xtrain[i:i+self.batch_size]
                y_pred = self.model(Xbatch)
                ybatch = Ytrain[i:i+self.batch_size]
                loss = self.loss_fn(y_pred, ybatch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch >=0: #== 0 or epoch == self.n_epochs:
                print(f'Finished epoch {epoch}, latest loss {loss}')

        # compute accuracy (no_grad is optional)
        #
        #     y_pred = model(X2)

    def predict(self,X):
        with torch.no_grad():
            Y= self.model(X)
        return Y
    




if __name__ == "__main__":
    from configs.config import get_cfg
    cfg = get_cfg("model_finance.yaml")
    # Instantiate and print the model
    model = MLP(cfg.model, (90, 3))
    print(model.model)