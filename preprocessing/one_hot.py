import numpy as np

# Performs one hot encoding of it's inputs
def one_hot(inputs):
       # Create an array of zeros with dimensions of input_size x maximum_class_label
       # (need to add one to max due to zero class label )
       result = np.zeros((inputs.size, inputs.max() + 1))
       # For each row, set the value at column indexed by input to 1
       result[np.arange(inputs.size), inputs] = 1
       return result

if __name__ == "__main__":
       a = np.array([2, 0, 2, 0,1])
       print(one_hot(a))