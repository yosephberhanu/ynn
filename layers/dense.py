import numpy as np
class Dense:
	def __init__(self, n_inputs, n_neurons, l1 = 0, l2 = 0 ):
		# Initialize the weight matrix to random values of 
		# the dimension n_inputs x n_neurons
		self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
		# Initialize the bias vector with n_neurons dimensions
		self.bias = np.zeros((1, n_neurons))
		self.l1 = l1
		self.l2 = l2
	
	# Perform the forward pass
	def forward(self, inputs):
		# Remember input values for backpropagation
		self.inputs = inputs
		self.output = np.dot(inputs, self.weights) + self.bias
	
	# Perform the backward pass
	def backward(self, dvalues):
		# Gradients on parameters
		self.dweights = np.dot(self.inputs.T, dvalues)
		self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)
		# L1 Regularization
		if self.l1 > 0:
			# L1 on weights
			dL1 = np.ones_like(self.weights)
			dL1[self.weights < 0] = -1
			self.dweights += self.l1 * dL1
			# L1 on bias
			dL1 = np.ones_like(self.bias)
			dL1[self.bias < 0] = -1
			self.dbiases += self.l1 * dL1
		# L2 Regularization

		if self.l2 > 0:
			# L2 on weights
			self.dweights += 2 * self.l2 * self.weights
			# L2 on bias
			self.dbiases += 2 * self.l2 * self.bias
		
		# Gradient on values
		self.dinputs = np.dot(dvalues, self.weights.T)