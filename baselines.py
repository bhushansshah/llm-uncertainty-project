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

def trace_length(tokens: list):
    """
    Compute the uncertainty as the length of the trace.

    Args:
        tokens (list): A list of tokens

    Returns:    
        float: The uncertainty as the length of the trace
    """
    return len(tokens)

def normalized_trace_length(tokens: list, mean_length: float, std_length: float):
    """
    Compute the normalized length of the trace.
    The normalized length is the trace length minus the mean length divided by the standard deviation of the trace length.
    Args:
        tokens (list): A list of tokens
        mean_length (float): The mean length of the trace
        std_length (float): The standard deviation of the trace length

    Returns:
        float: The normalized trace length
    """
    return (trace_length(tokens) - mean_length) / std_length
    
def num_forking_tokens(tokens: list, forking_tokens: set):
    """
    Compute the number of forking tokens.
    Args:
        tokens (list): A list of tokens
        forking_tokens (set): A set of forking tokens
    Returns:
        int: The number of forking tokens
    """
    forking_tokens_count = 0
    for token in tokens:
        if token in forking_tokens:
            forking_tokens_count += 1
    return forking_tokens_count
