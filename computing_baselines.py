import argparse
import pandas as pd
import numpy as np
from utils import load_results, compute_auroc, get_data_config, token_entropy
from baselines import average_logprobs, average_token_entropy, trace_length, num_forking_tokens, prob_answer_token, prob_answer_token_scifact
import os
from transformers import AutoTokenizer
from collections import defaultdict

def compute_neg_avg_logprobs(results):
    """
    Compute negative average log-probability for each result.

    Args:
        results (list[dict]): Loaded result dicts for one dataset/model.

    Returns:
        list[float]: Negative average log-probabilities.
    """
    neg_avg_logprobs = []
    for item in results:
        logprobs = item["response"]["logprobs"]["token_logprobs"]
        avg_logprobs = average_logprobs(logprobs)
        neg_avg_logprobs.append(-avg_logprobs)
    return neg_avg_logprobs

def compute_avg_token_entropy(results):
    """
    Compute the average token entropy for each result.

    Args:
        results (list[dict]): Loaded result dicts for one dataset/model.

    Returns:
        float: The average token entropy.
    """
    token_entropies = []
    for item in results:
        logprobs = item["response"]["logprobs"]["top_logprobs"]
        avg_entropy = average_token_entropy(logprobs)   
        token_entropies.append(avg_entropy)
    return token_entropies

def compute_uncertainty_as_trace_length(results: list[dict]):
    """
    Compute the uncertainty as the length of the trace.

    Args:
        results (list[dict]): Loaded result dicts for one dataset/model.

    Returns:
        list[float]: The uncertainty as the length of the trace
    """
    trace_lengths = []
    for item in results:
        tokens = item["response"]["logprobs"]["tokens"]
        trace_len = trace_length(tokens)
        trace_lengths.append(trace_len)
    return trace_lengths

def compute_num_forking_tokens(results: list[dict], forking_tokens_set: set):
    """
    Compute the number of forking tokens.
    Args:
        results (list[dict]): Loaded result dicts for one dataset-model pair.
        forking_tokens_set (set): A set of forking tokens
    Returns:
        list[int]: The number of forking tokens for each data point.
    """
    num_forking_tokens_list = []
    for item in results:
        tokens = item["response"]["logprobs"]["tokens"]
        num_forking_tokens_count = num_forking_tokens(tokens, forking_tokens_set)
        num_forking_tokens_list.append(num_forking_tokens_count)
    return num_forking_tokens_list

def compute_uncertainty_as_answer_prob(results: list[dict], model_name:str, dataset:str):
    """
    Compute the uncertainty as the probability of the answer token.

    Args:
        results (list[dict]): Loaded result dicts for one dataset/model.
        model_name (str): The name of the model.
        dataset (str): The name of the dataset.
    Returns:
        list[float]: The uncertainty as the probability of the answer token
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    answer_probs = []
    selected_examples_flag = []
    for item in results:
        tokens = item["response"]["logprobs"]["tokens"]
        top_logprobs = item["response"]["logprobs"]["top_logprobs"]
        selected_option = item["extracted_answer"]
        if dataset == "scifact_with_evidence" or dataset == "scifact_without_evidence":
            answer_prob = prob_answer_token_scifact(tokens, top_logprobs, selected_option, tokenizer)
        else:
            selected_option = item["extracted_answer"]
            answer_prob = prob_answer_token(tokens, top_logprobs, selected_option, tokenizer)
        if answer_prob is not None:
            answer_probs.append(answer_prob)
            selected_examples_flag.append(1)
        else:
            selected_examples_flag.append(0)
    return answer_probs, selected_examples_flag

def get_forking_tokens_set(results: list[dict]):
    """
    Get the set of forking tokens.
    Args:
        results (list[dict]): Loaded result dicts for one dataset-model pair.
    Returns:
        set: A set of forking tokens
    """
    token_entropies_dict = defaultdict(list)
    for item in results:
        tokens = item["response"]["logprobs"]["tokens"]
        tokens_top_logprobs = item["response"]["logprobs"]["top_logprobs"]
        for token, top_logprob in zip(tokens, tokens_top_logprobs):
            token_entropy_value = token_entropy(top_logprob)
            token_entropies_dict[token].append(token_entropy_value)
    
    avg_token_entropies_dict = defaultdict(float)
    for token, token_entropies in token_entropies_dict.items():
        if len(token_entropies) > 20:
            avg_token_entropy = np.mean(token_entropies)
            avg_token_entropies_dict[token] = avg_token_entropy
    avg_token_entropies_list = list(avg_token_entropies_dict.items())
    avg_token_entropies_list.sort(key=lambda x: x[1], reverse=True)
    forking_tokens_set = set([token for token, _ in avg_token_entropies_list[:50]])
    return forking_tokens_set

uncertainty_func_mapping = {
    "avg_logprobs": compute_neg_avg_logprobs,
    "avg_token_entropy": compute_avg_token_entropy,
    "trace_length": compute_uncertainty_as_trace_length,
    "forking_tokens": compute_num_forking_tokens,
    "answer_prob": compute_uncertainty_as_answer_prob,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run average log-probability baseline and compute AUROC."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="Names of the datasets (e.g. gpqa, mmlupro).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="List of model names to evaluate.",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        required=True,
        help="List of baselines to compute.",
        choices=["avg_logprobs", "avg_token_entropy", "trace_length", "forking_tokens", "answer_prob"],
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="outputs",
        help="Path to the outputs directory (default: outputs).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to save the results directory.",
    )
    args = parser.parse_args()

    
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    for baseline in args.baselines:
        rows = []
        for dataset in args.datasets:
            for model in args.models:
                print(f"Processing dataset={dataset}, model={model} for baseline={baseline} ...")
                results = load_results(args.outputs_dir, dataset, model)
                print(f"  Loaded {len(results)} examples.")
                if len(results) == 0:
                    print(f"  No results found for dataset={dataset}, model={model}")
                    continue
                uncertainty_func = uncertainty_func_mapping[baseline]
                if baseline == "answer_prob":
                    config = get_data_config(args.outputs_dir, dataset, model)
                    model = config["arguments"]["model"]
                    uncertainty_values, selected_examples_flag = uncertainty_func(results, model, dataset)
                    is_correct = []
                    for i, item in enumerate(results):
                        if selected_examples_flag[i] == 1:
                            is_correct.append(int(item["is_correct"]))
                    accuracy = np.mean(is_correct)
                    auroc = compute_auroc(uncertainty_values, is_correct)
                    rows.append(
                        {"dataset": dataset, "model": model, "auroc": round(auroc, 4), "accuracy": round(accuracy, 4), "num_selected_examples": len(is_correct)} 
                    )
                    print(f"  AUROC = {auroc:.4f}")
                    continue
                elif baseline == "forking_tokens":
                    forking_tokens_set = get_forking_tokens_set(results)
                    uncertainty_values = uncertainty_func(results, forking_tokens_set)
                    is_correct = [int(item["is_correct"]) for item in results]
                else:
                    uncertainty_values = uncertainty_func(results)
                    is_correct = [int(item["is_correct"]) for item in results]
                accuracy = np.mean([int(item["is_correct"]) for item in results])
                auroc = compute_auroc(uncertainty_values, is_correct)
                print(f"  AUROC = {auroc:.4f}")

                rows.append(
                    {"dataset": dataset, "model": model, "auroc": round(auroc, 4), "accuracy": round(accuracy, 4)}
                )
        # Save results
        csv_path = os.path.join(results_dir, f"{baseline}_baselines.csv")
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"\nResults for baseline={baseline} saved to {csv_path}")
    print(f"\nAll results saved to {results_dir}")

if __name__ == "__main__":
    main()
