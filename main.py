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

activation2 = Softmax()

loss = CategoricalCrossEntropy()

layer1.forward(X)

activation1.forward(layer1.output)

layer2.forward(activation1.output)

activation2.forward(layer2.output)

print(loss.calculate(activation2.output, y))
# This also works 
# print(loss.calculate(activation2.output, one_hot(y)))
