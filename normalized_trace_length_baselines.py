import argparse
import pandas as pd
import numpy as np
from utils import load_results, compute_auroc
from baselines import trace_length, normalized_trace_length

def compute_normalized_uncertainty_as_trace_length(results: list[dict]):
    """
    Compute the normalized uncertainty as the length of the trace.

    Args:
        results (list[dict]): Loaded result dicts for one dataset/model.

    Returns:
        list[float]: The normalized uncertainty as the length of the trace
    """
    trace_lengths = []
    for item in results:
        tokens = item["response"]["logprobs"]["tokens"]
        trace_len = trace_length(tokens)
        trace_lengths.append(trace_len)
    
    mean_trace_length = np.mean(trace_lengths)
    std_trace_length = np.std(trace_lengths)
    normalized_trace_lengths = []
    for item in results:
        tokens = item["response"]["logprobs"]["tokens"]
        normalized_trace_len = normalized_trace_length(tokens, mean_trace_length, std_trace_length)
        normalized_trace_lengths.append(normalized_trace_len)
    return normalized_trace_lengths

def main():
    parser = argparse.ArgumentParser(
        description="Run normalized trace length baseline and compute AUROC."
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
            #print one example question and answer
            normalized_trace_lengths = compute_normalized_uncertainty_as_trace_length(results)
            is_correct = [int(item["is_correct"]) for item in results]
            auroc = compute_auroc(normalized_trace_lengths, is_correct)
            print(f"  AUROC = {auroc:.4f}")

            rows.append(
                {"dataset": dataset, "model": model, "auroc": round(auroc, 4)}
            )

    # Save results
    csv_path = args.results_filepath
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
