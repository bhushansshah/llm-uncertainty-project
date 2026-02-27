import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from baselines import (
    average_logprobs,
    average_token_entropy,
    normalized_num_forking_tokens,
    num_forking_tokens,
    prob_answer_token,
    prob_answer_token_scifact,
    trace_length,
)
from utils import compute_auroc, get_data_config, load_results, token_entropy


def extract_logprob_views(item: dict):
    """
    Return (tokens, token_logprobs, top_logprobs_dicts) for both old and verbalized formats.

    Old format:
      response.logprobs.tokens
      response.logprobs.token_logprobs
      response.logprobs.top_logprobs (list[dict[str, float]])

    Verbalized format:
      response.logprobs.content (list[dict]) where each entry has:
      token, logprob, top_logprobs (list[dict{token, logprob}])
    """
    response = item.get("response", {})
    logprobs = response.get("logprobs", {})

    if isinstance(logprobs.get("content"), list):
        tokens = []
        token_logprobs = []
        top_logprobs = []

        for token_entry in logprobs["content"]:
            if not isinstance(token_entry, dict):
                continue

            token = token_entry.get("token")
            if token is None:
                continue

            logprob = token_entry.get("logprob", float("-inf"))
            candidates = token_entry.get("top_logprobs", [])

            top_logprobs_dict = {}
            if isinstance(candidates, list):
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    cand_token = candidate.get("token")
                    cand_logprob = candidate.get("logprob")
                    if cand_token is None or cand_logprob is None:
                        continue
                    top_logprobs_dict[cand_token] = cand_logprob
            elif isinstance(candidates, dict):
                top_logprobs_dict = candidates

            if token not in top_logprobs_dict and logprob is not None:
                top_logprobs_dict[token] = logprob

            tokens.append(token)
            token_logprobs.append(logprob)
            top_logprobs.append(top_logprobs_dict)

        return tokens, token_logprobs, top_logprobs

    tokens = logprobs.get("tokens", [])
    token_logprobs = logprobs.get("token_logprobs", [])
    top_logprobs = logprobs.get("top_logprobs", [])
    return tokens, token_logprobs, top_logprobs


def compute_neg_avg_logprobs(results):
    neg_avg_logprobs = []
    for item in results:
        _, token_logprobs, _ = extract_logprob_views(item)
        avg_logprobs = average_logprobs(token_logprobs)
        neg_avg_logprobs.append(-avg_logprobs)
    return neg_avg_logprobs


def compute_avg_token_entropy(results):
    token_entropies = []
    for item in results:
        _, _, top_logprobs = extract_logprob_views(item)
        avg_entropy = average_token_entropy(top_logprobs)
        token_entropies.append(avg_entropy)
    return token_entropies


def compute_uncertainty_as_trace_length(results: list[dict]):
    trace_lengths = []
    for item in results:
        tokens, _, _ = extract_logprob_views(item)
        trace_lengths.append(trace_length(tokens))
    return trace_lengths


def compute_num_forking_tokens(results: list[dict], forking_tokens_set: set):
    num_forking_tokens_list = []
    for item in results:
        tokens, _, _ = extract_logprob_views(item)
        num_forking_tokens_count = num_forking_tokens(tokens, forking_tokens_set)
        num_forking_tokens_list.append(num_forking_tokens_count)
    return num_forking_tokens_list


def compute_normalized_num_forking_tokens(results: list[dict], forking_tokens_set: set):
    num_forking_tokens_list = []
    for item in results:
        tokens, _, _ = extract_logprob_views(item)
        if len(tokens) == 0:
            num_forking_tokens_list.append(0.0)
            continue
        num_forking_tokens_count = normalized_num_forking_tokens(tokens, forking_tokens_set)
        num_forking_tokens_list.append(num_forking_tokens_count)
    return num_forking_tokens_list


def compute_uncertainty_as_answer_prob(results: list[dict], model_name: str, dataset: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    answer_probs = []
    selected_examples_flag = []
    for item in results:
        tokens, _, top_logprobs = extract_logprob_views(item)
        selected_option = item.get("extracted_answer")
        if selected_option is None:
            selected_examples_flag.append(0)
            continue
        if dataset == "scifact_with_evidence" or dataset == "scifact_without_evidence":
            answer_prob = prob_answer_token_scifact(tokens, top_logprobs, selected_option, tokenizer)
        else:
            answer_prob = prob_answer_token(tokens, top_logprobs, selected_option, tokenizer)
        if answer_prob is not None:
            answer_probs.append(answer_prob)
            selected_examples_flag.append(1)
        else:
            selected_examples_flag.append(0)
    return answer_probs, selected_examples_flag


def get_forking_tokens_set(results: list[dict]):
    token_entropies_dict = defaultdict(list)
    for item in results:
        tokens, _, tokens_top_logprobs = extract_logprob_views(item)
        for token, top_logprob in zip(tokens, tokens_top_logprobs):
            if not isinstance(top_logprob, dict) or len(top_logprob) == 0:
                continue
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

def compute_verbalized_uncertainty(results: list[dict]):
    verbalized_class_to_confidence = {
        "almost no chance": 0.05,
        "highly unlikely": 0.15,
        "chances are slight": 0.25,
        "unlikely": 0.35,
        "less than even": 0.45,
        "better than even": 0.55,
        "likely": 0.65,
        "very good chance": 0.75,
        "highly likely": 0.85,
        "almost certain": 0.95,
    }
    uncertainty_scores = []
    selected_examples_flag = []
    for item in results:
        confidence = item.get("verbalized_confidence", None)
        is_correct = item.get("is_correct", None)
        if confidence is None or is_correct is None:
            selected_examples_flag.append(0)
            continue

        if isinstance(confidence, str):
            confidence = verbalized_class_to_confidence.get(confidence.strip().lower())
            if confidence is None:
                selected_examples_flag.append(0)
                continue
        else:
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                selected_examples_flag.append(0)
                continue

        confidence = max(0.0, min(1.0, confidence))
        uncertainty_scores.append(1.0 - confidence)
        selected_examples_flag.append(1)
    return uncertainty_scores, selected_examples_flag

uncertainty_func_mapping = {
    "avg_logprobs": compute_neg_avg_logprobs,
    "avg_token_entropy": compute_avg_token_entropy,
    "trace_length": compute_uncertainty_as_trace_length,
    "forking_tokens": compute_num_forking_tokens,
    "normalized_forking_tokens": compute_normalized_num_forking_tokens,
    "answer_prob": compute_uncertainty_as_answer_prob,
    "verbalized": compute_verbalized_uncertainty,
}

def rename_keys(result, rename_map):
    """
    Rename keys in a result dictionary.
    Args:
        result (dict): A result dictionary.
        rename_map (dict): A dictionary of old keys to new keys.
    Returns:
        dict: A result dictionary with renamed keys.
    """
    return {rename_map[k]: v for k, v in result.items() if k in rename_map}

def main():
    parser = argparse.ArgumentParser(
        description="Compute AUROC baselines for verbalized output format."
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
        choices=[
            "avg_logprobs",
            "avg_token_entropy",
            "trace_length",
            "forking_tokens",
            "normalized_forking_tokens",
            "answer_prob",
            "verbalized",
        ],
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="outputs/verbalized",
        help="Path to verbalized outputs directory (default: outputs/verbalized).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to save the results directory.",
    )
    args = parser.parse_args()

    rename_map = {
        "model": "Model",
        "dataset": "Dataset",
        "accuracy": "Accuracy",
        "num_selected_examples": "Num Selected Examples",
        "avg_logprobs": "Avg. Neg Log Prob",
        "avg_token_entropy": "Token Entropy",
        "trace_length": "Trace Length",
        "forking_tokens": "Forking Tokens",
        "normalized_forking_tokens": "Normalized Forking Tokens",
        "answer_prob_accuracy": "Answer Probs accuracy",
        "answer_prob_num_selected_examples": "Answer Probs Num Selected Examples",  
        "answer_prob": "Answer Probs",
        "verbalized": "Verbalized",
    }
    results_list = []
    os.makedirs(args.results_dir, exist_ok=True)
    for dataset in args.datasets:
        for model in args.models:
            print(f"Loading results={dataset}, model={model} ...")
            results = load_results(args.outputs_dir, dataset, model)
            print(f"  Loaded {len(results)} examples.")
            if len(results) == 0:
                print(f"  No results found for dataset={dataset}, model={model}")
                continue
            baselines_result = {}
            for baseline in args.baselines:
                print(f"Computing {baseline} for dataset={dataset}, model={model} ...")
                uncertainty_func = uncertainty_func_mapping[baseline]
                if baseline == "verbalized":
                    uncertainty_values, selected_examples_flag = uncertainty_func(results)
                    is_correct = []
                    for i, item in enumerate(results):
                        if selected_examples_flag[i] == 1:
                            is_correct.append(int(item["is_correct"]))
                    if len(is_correct) == 0:
                        print("  No selected examples; skipping.")
                        continue
                    accuracy = np.mean(is_correct)
                    auroc = compute_auroc(uncertainty_values, is_correct)
                    baselines_result[baseline] = round(auroc, 4)
                    baselines_result["accuracy"] = round(accuracy, 4)
                    baselines_result["num_selected_examples"] = len(is_correct)
                    print(f"  AUROC = {auroc:.4f}")
                    continue

                if baseline == "answer_prob":
                    config = get_data_config(args.outputs_dir, dataset, model)
                    tokenizer_model = config["arguments"]["model"]
                    uncertainty_values, selected_examples_flag = uncertainty_func(
                        results, tokenizer_model, dataset
                    )
                    is_correct = []
                    for i, item in enumerate(results):
                        if selected_examples_flag[i] == 1:
                            is_correct.append(int(item["is_correct"]))
                    if len(is_correct) == 0:
                        print("  No examples matched answer-token extraction; skipping.")
                        continue
                    accuracy = np.mean(is_correct)
                    auroc = compute_auroc(uncertainty_values, is_correct)
                    baselines_result[baseline] = round(auroc, 4)
                    baselines_result["answer_prob_accuracy"] = round(accuracy, 4)
                    baselines_result["answer_prob_num_selected_examples"] = len(is_correct)
                    print(f"  AUROC = {auroc:.4f}")
                    continue

                if baseline == "forking_tokens" or baseline == "normalized_forking_tokens":
                    forking_tokens_set = get_forking_tokens_set(results)
                    uncertainty_values = uncertainty_func(results, forking_tokens_set)
                else:
                    uncertainty_values = uncertainty_func(results)

                is_correct = [int(item["is_correct"]) for item in results]
                accuracy = np.mean(is_correct)
                auroc = compute_auroc(uncertainty_values, is_correct)
                print(f"  AUROC = {auroc:.4f}")

                baselines_result[baseline] = round(auroc, 4)
                baselines_result["accuracy"] = round(accuracy, 4)
                baselines_result["num_selected_examples"] = len(is_correct)
                print(f"  AUROC = {auroc:.4f}")
                continue
            baselines_result['model'] = model
            baselines_result['dataset'] = dataset
            baselines_result = rename_keys(baselines_result, rename_map)
            results_list.append(baselines_result)
    # Save results
    csv_path = os.path.join(args.results_dir, f"verbalized_baselines.csv")
    # Load existing results into a DataFrame
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=rename_map.values())
    # Update existing results with new results or create new rows
    for result in results_list:
        mask = (df['Model'] == result['Model']) & (df['Dataset'] == result['Dataset'])
        if mask.any():
            df.loc[mask, list(result.keys())] = list(result.values())
        else:
            df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    # Sort rows so that all rows for the same model appear together
    df = df.sort_values(by=["Model", "Dataset"]).reset_index(drop=True)
    # Reorder columns to the desired order (others, if any, go to the end)
    preferred_internal_order = [
        "Model",
        "Dataset",
        "Accuracy",
        "Num Selected Examples",
        "Answer Probs accuracy",
        "Answer Probs Num Selected Examples",
        "Avg. Neg Log Prob",
        "Token Entropy",
        "Trace Length",
        "Forking Tokens",
        "Normalized Forking Tokens",
        "Answer Probs",
        "Verbalized",
    ]
    ordered_cols = [c for c in preferred_internal_order if c in df.columns] + [
        c for c in df.columns if not c in preferred_internal_order
    ]
    df = df[ordered_cols]
    df.to_csv(csv_path, index=False)
    print(f"\nAll results saved to {csv_path}")
if __name__ == "__main__":
    main()
