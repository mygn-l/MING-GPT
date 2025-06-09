import numpy as np
import math

BETA1 = 0.9
BETA2 = 0.98
EPSILON = 1 / 1000000000

def Adam(gradient, m_prev, v_prev, a_prev):
    m = BETA1 * m_prev + (1 - BETA1) * gradient
    v = BETA2 * v_prev + (1 - BETA2) * np.multiply(gradient, gradient)

    m_hat = m / (1 - BETA1)
    v_hat = v / (1 - BETA2)

    a = a_prev * math.sqrt(1 - BETA2) / (1 - BETA1)

    return gradient - (a / (math.sqrt(v_hat) + EPSILON)) * m_hat
