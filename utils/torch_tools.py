import torch
import torch.nn as nn
import torch.optim as optim
import copy 
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
        self.optimizer = optim.Adam(self.model.parameters())
        self.verbose = 0

    def train(self, Xtrain, Ytrain, Xvalid, Yvalid):
        # Initialize a counter for consecutive validation loss increases
        max_number_of_increase = 3
        valid_loss_increase = 0
        previous_loss_valid = float('inf')  # Set to infinity initially to ensure the first comparison passes
        saved_before_increase = copy.deepcopy(self.model)

        for epoch in range(self.n_epochs):
            print('Epoch: {}'.format(epoch))
            
            # Training and validation loss computation
            loss_train = self.epoch_train_set(Xtrain, Ytrain)
            loss_valid = self.epoch_valid_set(Xvalid, Yvalid)
            
            # Print the training and validation losses
            print('Train - Loss: {:.4f}'.format(loss_train))
            print('Valid - Loss: {:.4f}'.format(loss_valid))
            
            # Check if validation loss increased
            if loss_valid > previous_loss_valid:
                valid_loss_increase += 1  # Increment counter if validation loss increased
            else:
                valid_loss_increase = 0  # Reset counter if validation loss decreased or stayed the same
                saved_before_increase = copy.deepcopy(self.model)
            
            # Update the previous validation loss
            previous_loss_valid = loss_valid
            
            # Stop training if validation loss has increased 3 times consecutively
            if valid_loss_increase >= max_number_of_increase:
                print(f"Stopping early due to {max_number_of_increase} many consecutive increases in validation loss. Loading back earlier model.")
                self.model = copy.deepcopy(saved_before_increase)
                break
    
    def epoch_train_set(self, Xtrain, Ytrain):
        self.model.train()
        loss = 0
        for i in range(0, len(Xtrain), self.batch_size):
            Xbatch = Xtrain[i:i+self.batch_size]
            y_pred = self.model(Xbatch)
            ybatch = Ytrain[i:i+self.batch_size]
            self.optimizer.zero_grad()
            batch_loss =self.loss_fn(y_pred, ybatch)
            batch_loss.backward()
            self.optimizer.step()
            loss += batch_loss
        return loss
        

        # compute accuracy (no_grad is optional)
        #
        #     y_pred = model(X2)
    def epoch_valid_set(self,Xvalid,Yvalid):
        self.model.eval()
        loss = 0
        with torch.no_grad():
            for i in range(0, len(Xvalid), self.batch_size):
                Xbatch = Xvalid[i:i+self.batch_size]
                y_pred = self.model(Xbatch)
                ybatch = Yvalid[i:i+self.batch_size]
                loss +=self.loss_fn(y_pred, ybatch)
        return loss

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