import numpy as np

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
            self.biases.append(np.random.rand(dims[i + 1]) * 0.04)

            self.weight_m_prev.append(np.zeros((dims[i + 1], dims[i])))
            self.weight_v_prev.append(np.zeros((dims[i + 1], dims[i])))
            self.bias_m_prev.append(np.zeros((dims[i + 1])))
            self.bias_v_prev.append(np.zeros((dims[i + 1])))

    def forward_train(self, input_vector):
        self.activations = []

        current_vector = input_vector
        for i in range(len(self.weights)):
            self.activations.append(current_vector)
            current_vector = np.clip(self.weights[i] @ current_vector + self.biases[i], 0, None)
        self.activations.append(current_vector)

        return current_vector.copy()

    def backward(self, dC_dY):
        current_dC_dY = dC_dY
        for i in reversed(range(len(self.weights))):
            dY_dpre = (self.activations[i + 1] > 0).view("i1")
            dC_dPRE = current_dC_dY * dY_dpre
            dC_dW = np.outer(dC_dPRE, self.activations[i])
            current_dC_dY = self.weights[i].T @ dC_dPRE

            bias_adam = Adam(dC_dPRE, self.bias_m_prev[i], self.bias_v_prev[i])
            self.biases[i] += bias_adam[0]
            self.bias_m_prev[i] = bias_adam[1]
            self.bias_v_prev[i] = bias_adam[2]

            weight_adam = Adam(dC_dW, self.weight_m_prev[i], self.weight_v_prev[i])
            self.weights[i] += weight_adam[0]
            self.weight_m_prev[i] = weight_adam[1]
            self.weight_v_prev[i] = weight_adam[2]

        return current_dC_dY
