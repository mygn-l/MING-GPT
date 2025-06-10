import numpy as np

def Cross_Entropy_Back(expected_prob_dist, predicted_prob_dist):
    # Add 0.0001 to prevent division by zero
    return np.divide(expected_prob_dist, predicted_prob_dist + 0.0001) * -1
