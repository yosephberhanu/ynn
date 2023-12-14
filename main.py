import numpy as np
import nnfs
from nnfs.datasets import spiral_data


from loss import SoftmaxCategoricalCrossEntropy, CategoricalCrossEntropy, Accuracy
from layers import Dense
from activation import Softmax, ReLU
from preprocessing import one_hot
from optimizers import Adam

nnfs.init()
X, y = spiral_data(samples=100, classes=3)

layer1 = Dense(2,64)

activation1 = ReLU()

layer2 = Dense(64,3)

loss_scce = SoftmaxCategoricalCrossEntropy()

loss_accuracy = Accuracy()

optimizer = Adam(learning_rate = 0.05, decay = 5e-7)
for epoch in range(10001):
	layer1.forward(X)
	activation1.forward(layer1.output)
	layer2.forward(activation1.output)
	ls = loss_scce.forward(layer2.output, y)
	accuracy = loss_accuracy.calculate(loss_scce.output, y)
	
	if not epoch % 100:
		print(f'epoch: {epoch},' +
			  f'acc: {accuracy:.3f},' +
			  f'loss: {ls:.3f},' +
			  f'lr: {optimizer.current_learning_rate},'
			)
	loss_scce.backward(loss_scce.output, y)
	layer2.backward(loss_scce.dinputs)
	activation1.backward(layer2.dinputs)
	layer1.backward(activation1.dinputs)

	optimizer.pre_update_params()
	optimizer.update_params(layer1)
	optimizer.update_params(layer2)
	optimizer.post_update_params()