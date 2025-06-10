import numpy as np

from Regularizer import Normalize_Factor
from Adam import Adam

class Feedforward:
    def __init__(self, dims):
        self.weights = []
        self.biases = []
        self.dims = dims

        self.bias_m_prev = []
        self.bias_v_prev = []
        self.weight_m_prev = []
        self.weight_v_prev = []

        for i in range(len(dims) - 1):
            self.weights.append(np.random.rand(dims[i + 1], dims[i]) * 0.04 - 0.02)
            self.biases.append(np.random.rand(dims[i + 1]) * 0.04 - 0.02)

            self.weight_m_prev.append(np.zeros((dims[i + 1], dims[i])))
            self.weight_v_prev.append(np.zeros((dims[i + 1], dims[i])))
            self.bias_m_prev.append(np.zeros((dims[i + 1])))
            self.bias_v_prev.append(np.zeros((dims[i + 1])))

    def forward_train(self, input_vector):
        self.activations = []
        self.preactivations = []

        current_vector = input_vector.copy()
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

            bias_adam = Adam(dC_dPRE, self.bias_m_prev[i], self.bias_v_prev[i], LEARNING_RATE)
            self.biases[i] += bias_adam[0]
            self.bias_m_prev[i] = bias_adam[1]
            self.bias_v_prev[i] = bias_adam[2]

            weight_adam = Adam(dC_dW, self.weight_m_prev[i], self.weight_v_prev[i], LEARNING_RATE)
            self.weights[i] += weight_adam[0]
            self.weight_m_prev[i] = weight_adam[1]
            self.weight_v_prev[i] = weight_adam[2]

            current_dC_dY = np.matmul(np.transpose(self.weights[i]), dC_dPRE)
            current_dC_dY *= Normalize_Factor(current_dC_dY)
            # Normalize gradient

        return current_dC_dY
