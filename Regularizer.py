import numpy as np
import math

KEEP_PROB = 0.9

def Smooth_Label(dim, index, smoothing_factor):
    label = np.ones((dim)) * smoothing_factor
    label[index] = 1 - dim * smoothing_factor
    return label

def Normalize_Factor(tensor):
    sum = abs(np.sum(tensor))
    # Normalize only if greater than 1
    return 1 / sum if sum > 1 else 1

def Dropout(matrix):
    rand_matrix = np.random.rand(matrix.shape[0], matrix.shape[1])
    dropped_matrix = matrix.copy()
    dropped_matrix[rand_matrix > KEEP_PROB] = 0
    return dropped_matrix / KEEP_PROB
