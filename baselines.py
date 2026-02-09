import numpy as np

def average_logprobs(logprobs: list):
    """
    Compute the average log-probability across all tokens.

    Args:
        logprobs (list of float): A list of log-probabilities
    """
    if not logprobs:
        return float('-inf')
    return np.mean(logprobs).item()

