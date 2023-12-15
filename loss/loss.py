import numpy as np
# Common loss class
class Loss:
    # Regularization loss calculation
    def regularaization_loss(self, layer):
        # 0 by default
        r_loss = 0

        # L1 regularization
        if layer.l1 > 0:
            r_loss += layer.l1 * np.sum(np.abs(layer.weights))
            r_loss += layer.l1 * np.sum(np.abs(layer.bias))
        # L2 regularization
        if layer.l2 > 0:
            r_loss += layer.l2 * np.sum(layer.weights * layer.weights)
            r_loss += layer.l2 * np.sum(layer.bias * layer.bias)
        return r_loss

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss