import numpy as np
class Dropout:
	def __init__(self, rate = 0 ):
		self.rate = 1 - rate
			
	# Perform the forward pass
	def forward(self, inputs):
		# Remember input values for backpropagation
		self.inputs = inputs
		self.binary_mask = np.random.binomial(1, \
			self.rate, size=inputs.shape) / self.rate
		self.output = inputs * self.binary_mask
	
	# Perform the backward pass
	def backward(self, dvalues):
		# Gradients on parameters
		self.dinputs = dvalues * self.binary_mask