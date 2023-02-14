import chochin.nn

import numpy as np



class ActivationFunction():
  def __init__(self, requires_grad=True):
    '''requires_grad: Necessarily required to be True for backpropagation.'''
    super(ActivationFunction,self).__init__()
    self.__parameters = []
    self.requires_grad = requires_grad
  
  def parameters(self):
    return self.__parameters
  
  def __setattr__(self, name,value):
    super().__setattr__(name,value)
    if not callable(value) and name!='_ActivationFunction__parameters':
      self.__parameters.append((name,value))
  
  def __call__(self,input):
    if not isinstance(input, np.ndarray):
      raise TypeError("Input must be of type numpy nd.array.")
    if not hasattr(self,'function'):
      raise NameError("The function has not been defined for this activation funcion.")
    if self.requires_grad:
      self.input = input
    return np.array([self.function(e) for e in input])
  
  def backward(self,front_layer_grads):
    if not self.requires_grad:
      raise RuntimeError("requires_grad set to False. Cannot calculate gradients.")
    if not hasattr(self,'derivative_function'):
      raise NameError("derivative_function not set for this activation function. Cannot calculate gradients.")
    return np.array([self.derivative_function(e) for e in self.input]).reshape(-1,1)*front_layer_grads.reshape(-1,1)
    
  def __repr__(self):
    return self.__class__.__name__ + "()"

  
  
  
class ReLU(ActivationFunction):
  def __init__(self,requires_grad=True):
    super(ReLU, self).__init__(requires_grad)
    self.function = lambda x: 0 if x<0 else x
    self.derivative_function = lambda x: 0 if x<0 else 1

    
    
    
class Sigmoid(ActivationFunction):
  def __init__(self, requires_grad=True):
    super(Sigmoid,self).__init__(requires_grad)
    self.function = lambda x: 1/(1+np.exp(-x))
    self.derivative_function = lambda x: self.function(x)*(1-self.function(x))
    
    
    
    
    
    
    
