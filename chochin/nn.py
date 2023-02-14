from chochin.activations import ActivationFunction

import numpy as np


##################################---------LAYERS

class Linear():
  '''A Linear Layer'''
  def __init__(self,in_features,out_features, bias=True, requires_grad=True):
    self.in_features = in_features
    self.out_features = out_features
    self.requires_grad = requires_grad
    
    self.weights = np.random.rand(self.out_features,self.in_features)
    if bias:
      self.bias = np.random.rand(self.out_features)
    else:
      self.bias = False
  
  def forward(self,input):
    if not isinstance(input, np.ndarray):
      raise TypeError("Input must be a numpy nd.array.")
    if self.weights.shape[1] != input.shape[0]:
      raise ValueError("Cannot multiply matrix of dimension ",self.weights.shape," with input of dimension ",input.shape)
    
    if self.requires_grad:
      self.input = input
    
    result = np.dot(self.weights, input)
    if isinstance(self.bias, np.ndarray):
      result += self.bias
    return result
  
  
  def deriv_wrt_weight(self):
    if not hasattr(self,'input'):
      raise RuntimeError('backward() called before forward()')
    return self.input #previous layer h
  
  def deriv_wrt_bias(self):
    return 1
  
  def deriv_wrt_input(self):
    return self.weights
  
  def backward(self, front_layer_grads):
    if self.requires_grad:
      self.weight_grads = self.deriv_wrt_weight() * front_layer_grads #chain rule
      self.bias_grads = (self.deriv_wrt_bias() * front_layer_grads).reshape(1,-1)
    return np.dot(front_layer_grads.transpose(),self.deriv_wrt_input()) #chain rule for the back layer grads
  
  def get_weight_grads(self):
    if not self.requires_grad:
      raise RuntimeError('requires_grad set to False. Cannot calculate gradients.')
    if not hasattr(self,'weight_grads'):
      raise RuntimeError('get_weight_grads() called before backward()')
    return self.weight_grads
  
  def get_bias_grads(self):
    if not self.requires_grad:
      raise RuntimeError('requires_grad set to False. Cannot calculate gradients.')
    if not hasattr(self,'bias_grads'):
      raise RuntimeError('get_bias_grads() called before backward()')
    return self.bias_grads
  
  def get_grads(self):
    return {'weights':self.get_weight_grads(), 'bias':self.get_bias_grads(), 'requires_grad':self.requires_grad}
  
  
  def __call__(self,input):
    #overloading to use module as a function
    return self.forward(input)
  
  def __repr__(self):
    return f'Linear(in_features={self.in_features}, out_features={self.out_features}, bias={True if isinstance(self.bias,np.ndarray) else False})'
  
  def parameters(self):
    return {'weights':self.weights, 'bias':self.bias}

  
  
  
  
  
    
###################################################-----NEURAL NETWORK

class NeuralNetwork():
  def __init__(self):
    # super(NeuralNetwork, self).__init__()
    self.__layers = []
  
  def __setattr__(self, name,value):
    if isinstance(value,Linear) or isinstance(value,ActivationFunction):
      self.__layers.append((name,value))
    super().__setattr__(name,value)
  
  def __repr__(self):
    string = self.__class__.__name__ + "(\n"
    for name,layer in self.__layers:
      string += "  (" + name +"): " + layer.__repr__() + "\n"
    string += ")"
    return string
  
  def parameters(self):
    '''Doesnt return the references. To be used to just view the parameters.'''
    dicts = {}
    for name, layer in self.__layers:
      if not isinstance(layer,ActivationFunction):
        dicts[name] = layer.parameters()
    return dicts
  
  def param_references(self):
    '''To be used for updating the parameters.'''
    return self.__layers
  
  def backward(self,loss_function_grads):
    grads = loss_function_grads
    for name,layer in self.__layers[::-1]:
      grads = layer.backward(grads)
  
  def get_grads(self):
    dicts={}
    for name,layer in self.__layers:
      if not isinstance(layer,ActivationFunction):
        dicts[name]=layer.get_grads()
    return dicts
  
  def __call__(self,input):
    if not isinstance(input, np.ndarray):
      raise TypeError('Input must be of type numpy array.')
    ypred = np.array([])
    for i in range(len(input)):
      ypred = np.append(ypred, self.forward(input[i]))
    return ypred
  
  def forward(self,x):
    return x

  
