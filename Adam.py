import numpy as np

BETA1 = 0.9
BETA2 = 0.98
EPSILON = 0.000000001

def Adam(gradient, m_prev, v_prev, LEARNING_RATE):
    m = BETA1 * m_prev + (1 - BETA1) * gradient
    v = BETA2 * v_prev + (1 - BETA2) * np.multiply(gradient, gradient)

    m_hat = m / (1 - BETA1)
    v_hat = v / (1 - BETA2)

    return [(-LEARNING_RATE) * np.reciprocal(np.sqrt(v_hat) + EPSILON) * m_hat, m, v]
