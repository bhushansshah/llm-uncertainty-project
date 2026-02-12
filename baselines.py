import numpy as np
from utils import token_entropy
from transformers import AutoTokenizer

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

def prob_answer_token(tokens: list, top_logprobs:list[dict], selected_option: str):
    # find the index of "</" token
    if "</" in tokens:
        end_token_index = tokens.index("</")
    else:
        end_token_index = -1
    if end_token_index != -1:
        top_logprobs = top_logprobs[end_token_index:]
        tokens = tokens[end_token_index:]
    
    if "Answer" in tokens:
        answer_token_index = tokens.index("Answer")
    elif "answer" in tokens:
        answer_token_index = tokens.index("answer")
    else:
        answer_token_index = -1

    if answer_token_index == -1:
        if "Answer:" in tokens:
            answer_token_index = tokens.index("Answer:")
        elif "answer:" in tokens:
            answer_token_index = tokens.index("answer:")
        
        if answer_token_index != -1:
            option_token_ind = answer_token_index + 1
            option_token_top_logprobs = top_logprobs[option_token_ind]
            if selected_option in option_token_top_logprobs:
                return 1 - np.exp(option_token_top_logprobs[selected_option])
            else:
                return 0
        else:
            return None
    else:
        option_token_ind = answer_token_index + 2   
        option_token_top_logprobs = top_logprobs[option_token_ind]
        if selected_option in option_token_top_logprobs:
            return 1 - np.exp(option_token_top_logprobs[selected_option])
        else:
            return 0

def find_sublist(A, B):
    n, m = len(A), len(B)
    for i in range(m - n + 1):
        if B[i:i+n] == A:
            return i
    return -1

def prob_answer_token_scifact(tokens: list, top_logprobs:list[dict], selected_option: str, tokenizer:AutoTokenizer):

    tokens_string = "".join(tokens)
    selected_option = selected_option.upper()
    if selected_option not in tokens_string:
        print(f"Selected option {selected_option} not found in tokens string")
        return None
    print(f"Selected option ={selected_option}")
    selected_option_token_ids = tokenizer.encode(selected_option)
    selected_option_tokens = tokenizer.convert_ids_to_tokens(selected_option_token_ids)
    print(f"Selected option tokens = {selected_option_tokens}")
    # find the index of the selected option token ids in the tokens ie what is the start index of the selected option tokens in the tokens
    start_index = find_sublist(selected_option_tokens, tokens)
    if start_index == -1:
        print(f"Start index not found for selected option {selected_option} in tokens")
        return None
    else:
        total_logprob = 0.0
        for i, token in enumerate(selected_option_tokens):
            pos_top_logprobs = top_logprobs[start_index + i]
            if token in pos_top_logprobs:
                total_logprob += pos_top_logprobs[token]
            else:
                print(f"Token {token} not found in top logprobs")
                return None
        return 1 - np.exp(total_logprob)