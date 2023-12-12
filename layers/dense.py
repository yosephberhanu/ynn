import numpy as np
class Dense:
	def __init__(self, n_inputs, n_neurons ):
		# Initialize the weight matrix to random values of 
		# the dimension n_inputs x n_neurons
		self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
		# Initialize the bias vector with n_neurons dimensions
		self.bias = np.zeros((1, n_neurons))
	
	# Perform the forward pass
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.bias
