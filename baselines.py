from math import inf
import numpy as np
from utils import token_entropy, correct_space_tokens
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

def normalized_num_forking_tokens(tokens: list, forking_tokens: set):
    """
    Compute the normalized number of forking tokens.
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
    return forking_tokens_count / len(tokens)

def prob_answer_token(tokens: list, top_logprobs:list[dict], selected_option: str, tokenizer:AutoTokenizer):
    # find the index of "</" token
    variant1 = "Answer: " + selected_option
    variant2 = "answer: " + selected_option
    variant3 = "ANSWER: " + selected_option
    variant4 = "ANSWER " + selected_option
    variant5 = "answer " + selected_option
    variant6 = "Answer " + selected_option
    variant7 = " Answer: " + selected_option
    variant8 = " answer: " + selected_option
    variant9 = " ANSWER: " + selected_option
    variant10 = " ANSWER " + selected_option
    variant11 = " answer " + selected_option
    variant12 = " Answer " + selected_option
    variants = [variant1, variant2, variant3, variant4, variant5, variant6, variant7, variant8, variant9, variant10, variant11, variant12]
    selected_option_end_index = inf
    for variant in variants: 
        variant_token_ids = tokenizer.encode(variant)
        variant_tokens = tokenizer.convert_ids_to_tokens(variant_token_ids)
        variant_tokens = correct_space_tokens(variant_tokens)
        start_index = find_sublist(variant_tokens, tokens)
        if start_index != -1:
            selected_option_end_index = min(selected_option_end_index, start_index + len(variant_tokens) - 1)
    if selected_option_end_index == inf:
        print(f"Selected option {selected_option} not found in tokens for any variant. Returning None.")
        return None

    selected_option_token = tokens[selected_option_end_index]
    if selected_option_token not in top_logprobs[selected_option_end_index]:
        print(f"Selected option token {selected_option_token} not found in top logprobs at index {selected_option_end_index}. Returning None.")
        return None
    return 1 - np.exp(top_logprobs[selected_option_end_index][selected_option_token])

def find_sublist(A, B):
    n, m = len(A), len(B)
    for i in range(m - n + 1):
        if B[i:i+n] == A:
            return i
    return -1

def prob_answer_token_scifact(tokens: list, top_logprobs:list[dict], selected_option: str, tokenizer:AutoTokenizer):

    tokens_string = "".join(tokens)
    selected_option = selected_option.upper()
    if selected_option == "NEI":
        selected_option = "NOT ENOUGH INFO"
    
    if selected_option not in tokens_string:
        print(f"Selected option {selected_option} not found in tokens string")
        return None
    print(f"Selected option ={selected_option}")
    selected_options_variant1 = []
    selected_options_variant2 = []
    if selected_option == "SUPPORT":
        selected_options_variant1 = selected_option
        selected_options_variant2 = " SUPPORT"
    elif selected_option == "CONTRADICT":
        selected_options_variant1 = selected_option
        selected_options_variant2 = " CONTRADICT"
    elif selected_option == "NOT ENOUGH INFO":
        selected_options_variant1 = selected_option
        selected_options_variant2 = " NOT ENOUGH INFORMATION"
    else:
        print(f"Selected option {selected_option} not supported")
        return None
    
    selected_option_variant1_token_ids = tokenizer.encode(selected_options_variant1)
    selected_option_variant1_tokens = tokenizer.convert_ids_to_tokens(selected_option_variant1_token_ids)
    selected_option_variant1_tokens = correct_space_tokens(selected_option_variant1_tokens)
    selected_option_variant2_token_ids = tokenizer.encode(selected_options_variant2)
    selected_option_variant2_tokens = tokenizer.convert_ids_to_tokens(selected_option_variant2_token_ids)
    selected_option_variant2_tokens = correct_space_tokens(selected_option_variant2_tokens)
    start_index_variant1 = find_sublist(selected_option_variant1_tokens, tokens)
    start_index_variant2 = find_sublist(selected_option_variant2_tokens, tokens)
    if start_index_variant1 == -1 and start_index_variant2 == -1:
        print(f"Start index not found for selected option {selected_option} in tokens")
        return None
    
    selected_option_start_index = -1
    if start_index_variant1 != -1 and start_index_variant2 != -1:
        selected_option_start_index = min(start_index_variant1, start_index_variant2)
    elif start_index_variant1 != -1:
        selected_option_start_index = start_index_variant1
    elif start_index_variant2 != -1:
        selected_option_start_index = start_index_variant2
    selected_option_token = tokens[selected_option_start_index]
    if selected_option_token not in top_logprobs[selected_option_start_index]:
        print(f"Selected option token {selected_option_token} not found in top logprobs at index {selected_option_start_index}. Returning None.")
        return None
    return 1 - np.exp(top_logprobs[selected_option_start_index][selected_option_token])