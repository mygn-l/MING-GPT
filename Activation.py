import numpy as np

def Softmax(vector):
    exps = np.exp(vector)
    prob_dist = exps * (1 / np.sum(exps))
    return prob_dist

def Softmax_derivative(S):
    return np.diagflat(S) - np.outer(S, S)
