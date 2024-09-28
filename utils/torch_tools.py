import torch
import torch.nn as nn
import torch.optim as optim
import copy 
import collections



# Create the model class
class MLP:
    def __init__(self, archi, external_size):
        self.layers = nn.ModuleList()
        self.core = archi.layers.hidden_layers

        self.input_size = external_size[0] # updated input size
        self.output_size = external_size[1]  # updated output size

        self.layers.append(getattr(nn,archi.layers.input[0])(self.input_size, self.core.per_layer*self.input_size))  # updated input dimension
        self.layers.append(getattr(nn,archi.layers.input[1])())
        for i in range(self.core.count):
            self.layers.append(getattr(nn,self.core.type[0])(self.core.per_layer*self.input_size, self.core.per_layer*self.input_size))
            self.layers.append(getattr(nn,self.core.type[1])())
        self.layers.append(getattr(nn,archi.layers.output[0])(self.core.per_layer*self.input_size, self.output_size))  # updated output dimension
        if len(archi.layers.output)==2:
            self.layers.append(getattr(nn,archi.layers.output[1])())
        self.model = getattr(nn,archi.type)(*self.layers)


        self.loss_fn   = getattr(nn,archi.training.loss)()
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
    

class GeneralNN:
    def __init__(self, cfg):
        if cfg.type == "Sequential":
            self.model = build_seq_network(cfg.architecture)
        else:
            raise KeyError(f"The {cfg.type} cfgtecture has no been implemented.")

        self.loss_fn   = getattr(nn,cfg.training.loss)()
        self.n_epochs   = cfg.training.n_epochs
        self.batch_size   = cfg.training.batch_size
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

### The next two functions were taken from https://gist.github.com/ferrine/89d739e80712f5549e44b2c2435979ef
class Builder(object):
    def __init__(self, *namespaces):
        self._namespace = collections.ChainMap(*namespaces)

    def __call__(self, name, *args, **kwargs):
        try:
            return self._namespace[name](*args, **kwargs)
        except Exception as e:
            raise e.__class__(str(e), name, args, kwargs) from e

    def add_namespace(self, namespace, index=-1):
        if index >= 0:
            namespaces = self._namespace.maps
            namespaces.insert(index, namespace)
            self._namespace = collections.ChainMap(*namespaces)
        else:
            self._namespace = self._namespace.new_child(namespace)


def build_seq_network(architecture, builder=Builder(nn.__dict__)):
    """
    Configuration for feedforward networks is list by nature. We can write 
    this in simple data structures. In yaml format it can look like:

    .. code-block:: yaml

        architecture:
            - Conv2d:
                args: [3, 16, 25]
                stride: 1
                padding: 2
            - ReLU:
                inplace: true
            - Conv2d:
                args: [16, 25, 5]
                stride: 1
                padding: 2

    Note, that each layer is a list with a single dict, this is for readability.
    For example, `builder` for the first block is called like this:

    .. code-block:: python

        first_layer = builder("Conv2d", *[3, 16, 25], **{"stride": 1, "padding": 2})

    the simpliest ever builder is just the following function:

    .. code-block:: python

         def build_layer(name, *args, **kwargs):
            return layers_dictionary[name](*args, **kwargs)
    
    Some more advanced builders catch exceptions and format them in debuggable way or merge 
    namespaces for name lookup
    
    .. code-block:: python
    
        extended_builder = Builder(torch.nn.__dict__, mynnlib.__dict__)
        net = build_network(architecture, builder=extended_builder)
        
    """
    layers = []
    for block in architecture:
        assert len(block) == 1
        name, kwargs = list(block.items())[0]
        if kwargs is None:
            kwargs = {}
        args = kwargs.pop("args", [])
        layers.append(builder(name, *args, **kwargs))
    return nn.Sequential(*layers)

if __name__ == "__main__":
    # from configs.config import get_cfg
    # cfg = get_cfg("model_finance_old.yaml")
    # # Instantiate and print the model
    # model = MLP(cfg.model, (90, 3))
    # print(model.model)

    from configs.config import get_cfg
    cfg = get_cfg("model_finance.yaml")
    test = build_seq_network(cfg.model.architecture)
    print(test)