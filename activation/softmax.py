import numpy as np
import math
class Softmax:
	# Perform the SoftMax activation
	def forward(self, inputs):
		# Exponentiate, but first subtract the maximum value in each	
		# row to avoid the exponential overflow
		exp = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
		# Normalize to bring the output in the range of 0-1	
		# This can be interpreted as the confidence level on the classification
		self.output = exp / np.sum(exp, axis = 1, keepdims = True)
if __name__ == "__main__":
	sm = Softmax()
	sm.forward([[4.8,1.21,2.385],
			[8.9,-1.81,0.2],
			[1.41,1.051, 0.026]])
	print(sm.output)
