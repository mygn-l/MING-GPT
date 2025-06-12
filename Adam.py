import numpy as np

from config import LEARNING_RATE, EPSILON, BETA1, BETA2

def Adam(gradient, m_prev, v_prev):
    m = BETA1 * m_prev + (1 - BETA1) * gradient
    v = BETA2 * v_prev + (1 - BETA2) * (gradient * gradient)

    m_hat = m / (1 - BETA1)
    v_hat = v / (1 - BETA2)

    d_param = (-LEARNING_RATE) * np.reciprocal(np.sqrt(v_hat) + EPSILON) * m_hat

    return [d_param, m, v]
