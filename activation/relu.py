import numpy as np
class ReLU:
	# Perform the ReLU activation
	def forward(self, inputs):
		# Return zero for negative values, otherwise return as is
		self.output = np.maximum(0, inputs)


if __name__ == "__main__":
	a = np.array([0.2,-0.3,-0.4,0.5,-6,0.7, 4])
	relu = ReLU()
	relu.forward(a)
	print(relu.output)