import argparse
import pandas as pd
import numpy as np
from utils import load_results, compute_auroc
from baselines import average_logprobs

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
            #print one example question and answer
            neg_avg_logprobs = compute_neg_avg_logprobs(results)
            is_correct = [int(item["is_correct"]) for item in results]
            accuracy = np.mean([int(item["is_correct"]) for item in results])
            auroc = compute_auroc(neg_avg_logprobs, is_correct)
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
