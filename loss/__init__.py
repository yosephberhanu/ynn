if __name__ == "__main__":
	from crossentropy import CategoricalCrossEntropy
	from accuracy import Accuracy
	from mse import MeanSquaredError
	from mae import MeanAbsoluteError
else:
	from .crossentropy import CategoricalCrossEntropy
	from .accuracy import Accuracy
	from .mse import MeanSquaredError
	from .mae import MeanAbsoluteError