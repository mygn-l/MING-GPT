import numpy as np

class Feedforward:
    def __init__(self, dims):
        self.weights = []
        self.biases = []
        self.dims = dims

        for i in range(len(dims) - 1):
            self.weights.append(np.random.rand(dims[i + 1], dims[i]))
            self.biases.append(np.random.rand(dims[i + 1]))
    
    def forward_train(self, input_vector):
        self.activations = []
        self.preactivations = []

        current_vector = input_vector
        for i in range(len(self.weights)):
            self.activations.append(current_vector.copy())
            current_vector = np.matmul(self.weights[i], current_vector) + self.biases[i]
            self.preactivations.append(current_vector.copy())
            current_vector = np.clip(current_vector, 0, 1) #RELU capped  at 1
        self.activations.append(current_vector.copy())

        return current_vector

    def backward(self, dC_dY, LEARNING_RATE):
        current_dC_dY = dC_dY.copy()
        for i in reversed(range(len(self.weights))):
            d_RELU = (self.activations[i + 1] > 0).view('i1')
            dC_dPRE = current_dC_dY * d_RELU
            dC_dW = np.outer(dC_dPRE, self.activations[i])

            bruh = (1 / np.sum(dC_dPRE)) if np.sum(dC_dPRE) != 0 else 1
            self.biases[i] = self.biases[i] - dC_dPRE * LEARNING_RATE * abs(bruh)
            bruh = (1 / np.sum(dC_dW)) if np.sum(dC_dW) != 0 else 1
            self.weights[i] = self.weights[i] - dC_dW * LEARNING_RATE * abs(bruh)

            current_dC_dY = np.matmul(np.transpose(self.weights[i]), dC_dPRE)
            bruh = (1 / np.sum(current_dC_dY)) if np.sum(current_dC_dY) != 0 else 1
            current_dC_dY = current_dC_dY * abs(bruh)
            #NORMALIZED

        return current_dC_dY
