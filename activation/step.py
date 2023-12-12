import numpy as np
class Step:
	def __init__(self, threshold = 0.5):
		self.threshold = threshold
	# Perform the step funtion activation
	def forward(self, inputs):
		inputs[inputs >= self.threshold] = 1
		inputs[inputs < self.threshold] = 0
		self.output = inputs

if __name__ == "__main__":
	a = np.array([0.2,0.3,0.4,0.5,0.6,0.7])
	step = Step()
	step.forward(a)
	print(step.output)