import numpy as np




class LossFunction():
  def __init__(self, requires_grad=True):
    '''requires_grad: Necessarily True for backpropagation.'''
    self.__parameters = []
    self.requires_grad = requires_grad
    
  def __setattr__(self, name,value):
    if not callable(value) and name!='_LossFunction__parameters':
      self.__parameters.append((name,value))
    super().__setattr__(name,value)
  
  def parameters(self):
    return self.__parameters
  
  def __call__(self,yhat,y):
    if not isinstance(yhat, np.ndarray) or not isinstance(y, np.ndarray):
      raise TypeError("yhat and y both needs to be of type Numpy nd.array.")
    if self.requires_grad:
      self.yhat = yhat
      self.y = y
    return self.function(yhat,y)
  
  def backward(self):
    if not self.requires_grad:
      raise RuntimeError('requires_grad set to False. Cannot calculate gradients.')
    if not hasattr(self,'derivative_function'):
      raise NameError('derivative_function not set for this loss function. Cannot calculate gradients')
    return np.array([self.derivative_function(pred,gtruth) for pred,gtruth in zip(self.yhat, self.y)]).reshape(-1,1)

  
  
  
  
class MSELoss(LossFunction):
  def __init__(self, requires_grad=True):
    super(MSELoss, self).__init__(requires_grad)
    self.function = lambda yhat,y: np.square(yhat-y)
    self.derivative_function = lambda yhat,y: (yhat-y)

  def __call__(self, yhat,y):
    super().__call__(yhat,y)
    result = np.array([self.function(p,g) for p,g in zip(yhat,y)])
    result = result.sum()/len(result)
    return result
  

  
  
class BCELoss(LossFunction):
  '''Calculates loss wrt just one example.'''
  def __init__(self, requires_grad=True):
    super(BCELoss, self).__init__(requires_grad)
    self.function = lambda yhat,y:  (-y*np.log(yhat)-(1-y)*np.log(1-yhat))
    self.derivative_function = lambda yhat,y: (-y/yhat +(1-y)/(1-yhat))