import numpy as np


def pctc(arr:np.ndarray) -> np.ndarray: 
  """Calculate percentage change using first dimension of the 2D array"""
  return np.diff(arr,axis = 0) / arr[:-1,:] * 100