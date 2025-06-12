import numpy as np

def Cross_Entropy_Back(expected_prob_dist, predicted_prob_dist):
    return np.divide(expected_prob_dist, predicted_prob_dist + 0.001) * -1
