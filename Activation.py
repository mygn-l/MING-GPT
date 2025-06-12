import numpy as np

def Softmax(vectors):
    maxes = np.max(vectors, axis=1)
    dividend = np.exp(vectors - maxes.reshape((-1, 1)))
    divisor = np.sum(dividend, axis=1)
    return dividend / divisor.reshape((-1, 1))

def Softmax_derivative(S):
    return np.diagflat(S) - np.outer(S, S)
