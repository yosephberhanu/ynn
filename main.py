from loss import SoftmaxCategoricalCrossEntropy
from loss import CategoricalCrossEntropy
from layers import Dense
from activation import Softmax, ReLU
from preprocessing import one_hot
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()
X, y = spiral_data(samples=100, classes=3)

layer1 = Dense(2,3)

activation1 = ReLU()

layer2 = Dense(3,3)

scce = SoftmaxCategoricalCrossEntropy()

layer1.forward(X)

activation1.forward(layer1.output)

layer2.forward(activation1.output)

scce.forward(layer2.output, y)
print(scce.output[:5])


scce.backward(scce.output, y)
layer2.backward(scce.dinputs)
activation1.backward(layer2.dinputs)
layer1.backward(activation1.dinputs)

# Print gradients
print(layer1.dweights)
print(layer1.dbiases)
print(layer2.dweights)
print(layer2.dbiases)