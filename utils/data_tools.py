import numpy as np
import pandas as pd
import torch

class GenericArray:
    def __init__(self, data=None):
        """
        Initialize with a variety of data types: Pandas DataFrame, NumPy array, or PyTorch tensor.
        If no data is provided, initialize with an empty array.
        """
        if data is None:
            self.ndarray = np.array([])  # Initialize with an empty NumPy array
        elif isinstance(data, pd.DataFrame):
            self.ndarray = data.values  # Convert pandas DataFrame to NumPy array
        elif isinstance(data, np.ndarray):
            self.ndarray = data  # Already a NumPy array
        elif isinstance(data, torch.Tensor):
            self.ndarray = data.numpy()  # Convert PyTorch tensor to NumPy array
        else:
            raise TypeError("Unsupported input type. Use Pandas DataFrame, NumPy array, or PyTorch tensor.")
        
        self._tensor = None  # Cache for PyTorch tensor conversion

    @property
    def tensor(self):
        """Convert to a PyTorch tensor if not already cached."""
        if self._tensor is None:
            self._tensor = torch.from_numpy(self.ndarray.astype('float32'))
        return self._tensor

    def reshaped(self, shape=(-1, 1)):
        """Reshape the internal NumPy array."""
        return self.ndarray.reshape(shape)
    
    def reshaped_tensor(self, shape=(-1, 1)):
        """Reshape the internal NumPy array."""
        return self.tensor.reshape(*shape)
    
    def reshaped_2D_tensor(self):
        """returns the reshaped array in 2D, by flattening all other dimensions."""
        if self.ndarray.ndim <= 2:
            return self.tensor
        else:
            new_shape = (self.ndarray.shape[0], -1)
            return torch.from_numpy(self.ndarray.reshape(new_shape).astype('float32'))
