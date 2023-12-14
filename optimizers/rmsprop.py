import numpy as np
class RMSProp:
	def __init__(self, learning_rate = 1.0, decay = 0, epsilon = 1e-7, rho = 0.9):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.iterations = 0
		self.epsilon = epsilon
		self.rho = rho

	def pre_update_params(self):
		if self.decay:
			self.current_learning_rate = self.learning_rate * \
						(1. / (1. + self.decay * self.iterations))

	def update_params(self, layer):
		if not hasattr(layer, 'weight_cache'):
			layer.weight_cache = np.zeros_like(layer.weights)
			layer.bias_cache = np.zeros_like(layer.bias)
		
		layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) *\
							 layer.dweights ** 2
		layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) *\
							 layer.dbiases ** 2

		layer.weights += -self.current_learning_rate * layer.dweights /\
						(np.squrt(layer.weight_cache) + self.epsilon)
		layer.bias += -self.current_learning_rate * layer.dbiases /\
						(np.squrt(layer.bias_cache) + self.epsilon)

	def post_update_params(self):
		self.iterations += 1
