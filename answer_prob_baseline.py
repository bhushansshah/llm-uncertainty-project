import argparse
import pandas as pd
import numpy as np
from utils import load_results, compute_auroc
from baselines import prob_answer_token, prob_answer_token_scifact
from transformers import AutoTokenizer

def compute_uncertainty_as_answer_prob(results: list[dict], model:str, dataset:str):
    """
    Compute the uncertainty as the probability of the answer token.

    Args:
        results (list[dict]): Loaded result dicts for one dataset/model.

    Returns:
        list[float]: The uncertainty as the probability of the answer token
    """
    if model == "Qwen_Qwen3-32B":
        model_name = "Qwen/Qwen3-32B"
    elif model == "openai_gpt-oss-120b":
        model_name = "openai/gpt-oss-120b"
    else:
        raise ValueError(f"Model {model} not supported")
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
            selected_option = " " + item["extracted_answer"]
            answer_prob = prob_answer_token(tokens, top_logprobs, selected_option)
        if answer_prob is not None:
            answer_probs.append(answer_prob)
            selected_examples_flag.append(1)
        else:
            selected_examples_flag.append(0)
    return answer_probs, selected_examples_flag



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
            if len(results) == 0:
                print(f"  No results found for dataset={dataset}, model={model}")
                continue
            print(f"  Loaded {len(results)} examples.")
            answer_probs, selected_examples_flag = compute_uncertainty_as_answer_prob(results, model, dataset)
            is_correct = []
            for i, item in enumerate(results):
                if selected_examples_flag[i] == 1:
                    is_correct.append(int(item["is_correct"]))
            accuracy = np.mean(is_correct)
            auroc = compute_auroc(answer_probs, is_correct)


            rows.append(
                {"dataset": dataset, "model": model, "auroc": round(auroc, 4), "accuracy": round(accuracy, 4), "num_selected_examples": len(is_correct)} 
            )

            print(f"  AUROC = {auroc:.4f}")

    # Save results
    csv_path = args.results_filepath
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
