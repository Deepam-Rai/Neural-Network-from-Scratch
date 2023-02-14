from chochin.nn import NeuralNetwork
from chochin.activations import ActivationFunction

import numpy as np



class SGD():
  def __init__(self, model,lr):
    if not isinstance(model,NeuralNetwork):
      raise TypeError("Input type must be of type NeuralNetwork.")
    self.model = model
    self.lr = lr
    pass
  
  def step(self):
    for name, layer in self.model.param_references():
      if not isinstance(layer, ActivationFunction):
        if layer.requires_grad:
          layer.weights += - self.lr*layer.get_grads()['weights']
          if isinstance(layer.bias, np.ndarray):
            layer.bias += - self.lr*layer.get_grads()['bias'].flatten()
  def __repr__(self):
    s = f'\
    SGD ( \n\
      lr:{self.lr}\n\
    )'
    return s