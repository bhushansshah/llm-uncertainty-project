import argparse
import pandas as pd
import numpy as np
from utils import load_results, compute_auroc, token_entropy
from baselines import num_forking_tokens
from collections import defaultdict

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


def main():
    parser = argparse.ArgumentParser(
        description="Run forking tokens baseline and compute AUROC."
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
        "--outputs_dir",
        type=str,
        default="outputs",
        help="Path to the outputs directory (default: outputs).",
    )
    parser.add_argument(
        "--results_filepath",
        type=str,
        required=True,
        help="Path to save the results CSV.",
    )
    args = parser.parse_args()

    rows = []

    for dataset in args.datasets:
        for model in args.models:

            print(f"Processing dataset={dataset}, model={model} ...")
            results = load_results(args.outputs_dir, dataset, model)
            print(f"  Loaded {len(results)} examples.")
            if len(results) == 0:
                print(f"  No results found for dataset={dataset}, model={model}")
                continue
            forking_tokens_set = get_forking_tokens_set(results)
            num_forking_tokens_list = compute_num_forking_tokens(results, forking_tokens_set)
            is_correct = [int(item["is_correct"]) for item in results]
            accuracy = np.mean([int(item["is_correct"]) for item in results])
            auroc = compute_auroc(num_forking_tokens_list, is_correct)
            print(f"  AUROC = {auroc:.4f}")

            rows.append(
                {"dataset": dataset, "model": model, "auroc": round(auroc, 4), "accuracy": round(accuracy, 4)}
            )

    # Save results
    csv_path = args.results_filepath
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
