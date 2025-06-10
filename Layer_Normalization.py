import numpy as np

from Regularizer import Normalize_Factor
from Adam import Adam

EPSILON = 0.00001

class Layer_Normalization:
    def __init__(self, num_tokens):
        self.GAMMA = np.ones((num_tokens))
        self.BETA = np.zeros((num_tokens))
        self.EPSILON = EPSILON

        self.GAMMA_m_prev = np.zeros((num_tokens))
        self.GAMMA_v_prev = np.zeros((num_tokens))
        self.BETA_m_prev = np.zeros((num_tokens))
        self.BETA_v_prev = np.zeros((num_tokens))

    def forward_train(self, input_vectors):
        means = np.sum(input_vectors, axis=1) / input_vectors.shape[1]

        diff = input_vectors - means.reshape((-1, 1))
        self.diff = diff

        variance = np.sum(diff * diff, axis=1) / input_vectors.shape[1]

        dividing_factor = np.sqrt(variance + self.EPSILON)
        self.dividing_factor = dividing_factor

        pregamma = diff / dividing_factor.reshape((-1, 1))
        self.pregamma = pregamma

        output_vectors = pregamma * self.GAMMA.reshape((-1, 1))
        output_vectors += self.BETA.reshape((-1, 1))
        return output_vectors

    def backward(self, dC_dY, LEARNING_RATE):
        dC_dpregamma = dC_dY * self.GAMMA.reshape((-1, 1))

        dC_dprescaling = dC_dpregamma / self.dividing_factor.reshape((-1, 1))

        dC_dX = dC_dprescaling # Kronecker delta part
        dC_dX += np.sum(dC_dprescaling / (-dC_dY.shape[1]), axis=1).reshape((-1, 1)) # Mean part
        dC_dX += self.diff * (np.sum(dC_dY * self.diff, axis=1) * self.GAMMA / np.power(self.dividing_factor, 3) / (-dC_dY.shape[1])).reshape((-1, 1)) # Variance part

        dC_dgamma = np.sum(dC_dY * self.pregamma, axis=1)
        gamma_adam = Adam(dC_dgamma, self.GAMMA_m_prev, self.GAMMA_v_prev, LEARNING_RATE)
        self.GAMMA += gamma_adam[0]
        self.GAMMA_m_prev = gamma_adam[1]
        self.GAMMA_v_prev = gamma_adam[2]

        dC_dbeta = np.sum(dC_dY, axis=1)
        beta_adam = Adam(dC_dbeta, self.BETA_m_prev, self.BETA_v_prev, LEARNING_RATE)
        self.BETA += beta_adam[0]
        self.BETA_m_prev = beta_adam[1]
        self.BETA_v_prev = beta_adam[2]

        dC_dX *= Normalize_Factor(dC_dX)

        return dC_dX
