import numpy as np
class Linear:
	# Perform the linear activation function
	def forward(self, inputs):
		# Return the inputs as is
		self.output = inputs


if __name__ == "__main__":
	a = np.array([0.2,-0.3,-0.4,0.5,-6,0.7, 4])
	linear = Linear()
	linear.forward(a)
	print(linear.output)