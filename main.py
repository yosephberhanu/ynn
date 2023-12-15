import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from loss import SoftmaxCategoricalCrossEntropy, CategoricalCrossEntropy, Accuracy
from layers import Dense, Dropout
from activation import Softmax, ReLU
from preprocessing import one_hot
from optimizers import Adam

nnfs.init()
X, y = spiral_data(samples=100, classes=3)

layer1 = Dense(2,512, l2 = 5e-4)

activation1 = ReLU()

dropout = Dropout(0.1)

layer2 = Dense(512,3)

loss_scce = SoftmaxCategoricalCrossEntropy()

loss_accuracy = Accuracy()

optimizer = Adam(learning_rate = 0.05, decay = 5e-7)
for epoch in range(10001):
	layer1.forward(X)
	activation1.forward(layer1.output)
	dropout.forward(activation1.output)
	layer2.forward(dropout.output)
	ls = loss_scce.forward(layer2.output, y)
	accuracy = loss_accuracy.calculate(loss_scce.output, y)
	r_loss = loss_scce.loss.regularaization_loss(layer1)
	r_loss += loss_scce.loss.regularaization_loss(layer2)
	ls = ls + r_loss
	if not epoch % 100:
		print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {ls:.3f},' +
			  f'lr: {optimizer.current_learning_rate}')


	loss_scce.backward(loss_scce.output, y)
	layer2.backward(loss_scce.dinputs)
	dropout.backward(layer2.dinputs)
	activation1.backward(dropout.dinputs)
	layer1.backward(activation1.dinputs)

	optimizer.pre_update_params()
	optimizer.update_params(layer1)
	optimizer.update_params(layer2)
	optimizer.post_update_params()


#Validation
X_test, y_test = spiral_data(samples = 100, classes = 3)

layer1.forward(X_test)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
ls = loss_scce.forward(layer2.output, y_test)
accuracy = loss_accuracy.calculate(loss_scce.output, y_test)

print("Validation")
print(f'acc: {accuracy:.3f}, loss: {ls:.3f}')