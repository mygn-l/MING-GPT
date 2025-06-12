import numpy as np

from config import KEEP_PROB, SMOOTHING_FACTOR

def Smooth_Label(dim, index):
    label = np.ones((dim)) * SMOOTHING_FACTOR
    label[index] = 1 - dim * SMOOTHING_FACTOR
    return label

def Normalize_Factor(tensor):
    sum = abs(np.sum(tensor))
    # Normalize only if greater than 1
    return 1 / sum if sum > 1 else 1

def Dropout(matrix):
    rand_matrix = np.random.random(matrix.shape)
    dropped_matrix = matrix.copy()
    dropped_matrix[rand_matrix > KEEP_PROB] = 0
    return dropped_matrix / KEEP_PROB
