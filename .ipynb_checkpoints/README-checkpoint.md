# Chochin
This is a project to understand the neural network implementaion.  
The aim is to create a library similar to pytorch, with basic rudimentary functionalities.  

The library is named `chochin` and as of yet, following functionalities are present:
## Layers
`chochin.nn.Linear`
1. Linear Layer

## Activation Functions
`chochin.activations.<function>`
1. ReLU
2. Sigmoid

## Loss
`chochin.loss.<function>`
1. MSELoss()
2. BCELoss()

## Optimizers
`chochin.optim.<function>`
1. SGD

The demo.ipynb file shows the usage of the library.

# Hierarchy
<pre>
chochin: `import chochin`  
|---nn  
|   |---Linear  
|  
|---activations  
|   |---ReLU  
|   |---Sigmoid  
|  
|---loss  
|   |---MSELoss  
|   |---BCELoss  
|  
|---optim  
    |---SGD  
</pre>

# How to use.

1. In pytorch everything is in `torch.tensor` here everything is in `numpy.ndarray` if not error is raised.
2. Basic import: `import chochin`
3. Custom class: `class MyClass(chochin.nn.NeuralNetwork`
4. Layers: `self.hidden1 = chochin.nn.Linear(in_features=<some num>, out_features=<some num>, bias=<True or False>, requires_grad=<default True>`
5. Activation functions: `self.activation_fun = chochin.activations.ReLu()`
6. Loss functions: `loss_fun = chochin.loss.MSELoss()` which returns an object
7. Loss Calculation: `loss = loss_fun(yhat,y)` which returns a scalar value. **unlike pytorch**.
8. Calculating gradients: `model.backward(loss_fun.backward())` which doesnt return anything, **not like pytorch**
9. Optimizer: `optimizer = chochin.optim.SGD(model,lr=<some num>)`, where model is the `chochin.nn.NeuralNetwork` or its subclass object, **unlike model.parameters() in pytorch**.
10. Updating parameters: `optimizer.step()`
11. optim.zero_grads(): NA, everytime the gradients are overwritten.



# Implementation Details

## Neural Network Model
There is a base class `chochin.nn.NeuralNetwork`, and all further neural models should necessarily be subclass of it.  
It is equivalent to `torch.nn.Module`

## Layers
As of yet only linear layer is present: `chochin.nn.Linear` which is equivalent to `torch.nn.Linear`.  
It takes parameters:
- in_features : \<int>input features
- out_features : \<int> output features
- bias : \<boolean> if bias needs to be present or not
- requires_grad : \<boolean> whether gradient needs to be calculated or not

## Activation functions
There is a base class `ActivationFunction`, and all activation functions are necessarily its subclass.  
They take parameter `requires_grad = default True` because of the way gradients are calculated here naively. It needs to be necessarily True for backpropagation.

All activation functions necessarily needs to initialize:  
- `self.function: a->h`: calculate the output of provided aggregation value.
- `self.derivative_function: h wrt a`: used during backpropagation
  
  The available activation functions are:
  1. ReLU: `chochin.activations.ReLU(requires_grad=True)`
  2. Sigmoid: `chochin.activations.Sigmoid(requires_grad=True`

## Loss functions
Base class: `chochin.loss.LossFunction` and all loss functions necessarily need to be its subclass.

They take parameter `requires_grad = default True` which is required for backpropagation.

All subclass activation functions need to define following necessarily:
1. `self.function: yhat,y -> loss`
2. `self.derivative_function: yhat,y -> d(L)/d(yhat)`: Used during backpropagation.

Available Loss functions:  
1. `chochin.loss.MSELoss()`
2. `chochin.loss.BCELoss()`: Its a basic implementation and thus results are not so great.

## Optimizers
There is no base class for this yet.  
Available optimizers:
1. `chochin.optim.SGD()`: which takes `model` (instantiation of `chochin.nn.NeuralNetwork` or its subclass) and `lr`(learning rate) as parameters.


# Backpropagation calculation
1. Each layer has `derivative_function: inputs->derivative wrt inputs` or calculates the equivalent using functions.
2. Each layer stores the necessary inputs which are required to calculate the derivatives wrt some input.
	1. Because of this even activation functions and loss needs to store the inputs given to them.
3. Using the `derivative_function` (or equivalent) and the saved inputs, the partial derivative at the current node is calculated.
4. Further the partial derivatives is passed to back layers and just like in theory using the chain rule, product of partial derivatives are taken.



