import numpy as np
class MeanAbsoluteError:
	def calculate(self, y_pred, y_true):
		# Check if the y_true is one_hot encoded
		if len(y_true.shape) == 2:
			y_true = np.argmax(y_true, axis = 1)
		
		# Calculate average mean absolute error
		return np.mean(np.absolute(y_pred - y_true))
if __name__ == "__main__":
	loss = MeanAbsoluteError()
	y_pred = np.array([[0.7, 0.2, 0.1],
					   [0.5, 0.1, 0.4],
					   [0.02, 0.9, 0.08]])
	
	y_true = np.array([[1,0,0],[0,1,0],[0,1,0]])
	print("With one-hot encoding: ", loss.calculate(y_pred, y_true))
	
	y_true = np.array([0,1,1])
	print("Without one-hot encoding: ", loss.calculate(y_pred, y_true))
	
