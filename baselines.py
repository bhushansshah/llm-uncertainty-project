import numpy as np
from utils import token_entropy

def average_logprobs(logprobs: list):
    """
    Compute the average log-probability across all tokens.

    Args:
        logprobs (list of float): A list of log-probabilities
    """
    if not logprobs:
        return float('-inf')
    return np.mean(logprobs).item()

def average_token_entropy(token_logprobs: list[dict]):
    """
    Compute the average token entropy across all tokens.

    Args:
        token_logprobs (list of dict): A list of token log-probabilities
    
    Returns:
        float: The average token entropy, or -inf if the list is empty
    """
    # check if token_logprobs is a list of dicts
    if not isinstance(token_logprobs, list) or not all(isinstance(item, dict) for item in token_logprobs):
        raise ValueError("token_logprobs must be a list of dicts")
    token_entropies = []
    for logprob in token_logprobs:
        entropy = token_entropy(logprob)
        token_entropies.append(entropy)
    avg_token_entropy = average_logprobs(token_entropies)
    return avg_token_entropy