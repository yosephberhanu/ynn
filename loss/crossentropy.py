import numpy as np
from .loss import Loss

class CategoricalCrossEntropy(Loss):
	# Perform the forward pass
	def calculate(self, y_pred, y_true):
		# Clip the predicted values in y_pred to be in the range 1e-7, 1 - 1e-7
		# This ensures that our predictions are never zero which addresses the 
		# issue of log(0) not being defined
		y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
		entorpy = 0
		# Check if the y_true is one_hot encoded
		if len(y_true.shape) == 2:
			entorpy = np.sum(y_pred_clipped * y_true, axis=1)
		# Assuming y_true is not one_hot encoded
		else :
			entorpy = y_pred_clipped[np.arange(len(y_pred_clipped)), y_true]
		
		# Calculate average negative log of the entropy values
		return np.mean(-np.log(entorpy))
	
	def backward(self, dvalues, y_true):
		# Number of samples
		samples = len(dvalues)
		# Number of labels in every sample, We'll use the first sample to count them
		labels = len(dvalues[0])

		# If labels are sparse, turn them into one-hot vector
		if len(y_true.shape) == 1:
			y_true = np.eye(labels)[y_true]

		# Calculate gradient
		self.dinputs = -y_true / dvalues
		# Normalize gradient
		self.dinputs = self.dinputs / samples

if __name__ == "__main__":
	loss = CategoricalCrossEntropy()
	y_pred = np.array([[0.7, 0.1, 0.2],
					   [0.1, 0.5, 0.4],
					   [0.02, 0.9, 0.08]])
	
	y_true = np.array([[1,0,0],[0,1,0],[0,1,0]])
	print("With one-hot encoding: ", loss.calculate(y_pred, y_true))
	
	y_true = np.array([0,1,1])
	print("Without one-hot encoding: ", loss.calculate(y_pred, y_true))
	
