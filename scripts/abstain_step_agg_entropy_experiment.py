#!/usr/bin/env python3
"""
Step **cumulative (aggregated) entropy** abstention experiment (validation grid + test).

Same procedure as ``abstain_step_entropy_experiment.py``, but per-step uncertainty is the
**sum of mean chunk entropies** from chunk 0 through the current step (thinking-only token
entropies; non-overlapping chunks). See ``cumulative_sum_chunk_means`` in ``abstain_step_entropy``.

Hyperparameters (grid):
  - chunk_size: 50..700 step 50
  - delta: minimum gap between incorrect/correct **mean cumulative entropy** at a step on validation
    (0.1..10.0 step 0.3); same role as ``delta`` in ``build_active_and_tau``
  - noise: added to τ when comparing cumulative entropy at each step (1..30 step 0.5)
  - ground_threshold: min number of active steps exceeding τ+noise to abstain (1..50)

Active steps additionally require at least ``min_support_per_class`` correct **and** incorrect
validation responses at that step (default: 3), same as ``abstain_step_neg_logprob_experiment``.

Metrics and outputs mirror the step-entropy experiment; batch mode writes under
``abstaining_plots/<dataset>/agg_entropy/`` and ``abstaining_results/<dataset>/agg_entropy/``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from abstain_batch_utils import (  # noqa: E402
    METHOD_AGG_ENTROPY,
    discover_models_in_dataset,
)
from abstain_step_entropy import (  # noqa: E402
    abstention_f1,
    build_active_and_tau_with_min_support,
    chunk_step_means,
    cumulative_sum_chunk_means,
    filter_usable_examples_entropy,
    should_abstain,
    token_entropies_thinking_only,
    total_tokens_in_response,
    val_step_means_and_counts_from_step_lists,
)
from sklearn.model_selection import train_test_split  # noqa: E402


def load_results_flat_dir(results_dir: str) -> list[dict]:
    pattern = os.path.join(results_dir, "result_*.json")
    files = sorted(
        glob.glob(pattern),
        key=lambda f: int(os.path.basename(f).split("_")[1].split(".")[0]),
    )
    out = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def stratified_val_test_split(
    data: list[dict],
    val_size: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    n = len(data)
    if n <= val_size:
        raise ValueError(f"Need more than val_size={val_size} examples, got {n}")
    labels = [1 if d.get("is_correct") else 0 for d in data]
    idx = np.arange(n)

    use_stratify = len(set(labels)) >= 2 and val_size >= 2 and (n - val_size) >= 2
    if use_stratify:
        try:
            val_ix, test_ix = train_test_split(
                idx,
                train_size=val_size,
                random_state=seed,
                stratify=labels,
            )
            return [data[i] for i in val_ix], [data[i] for i in test_ix]
        except ValueError:
            pass

    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    val_data = [data[i] for i in idx[:val_size]]
    test_data = [data[i] for i in idx[val_size:]]
    return val_data, test_data


def evaluate_split_cached(
    step_series_per_example: list[list[float]],
    labels: list[bool],
    active: np.ndarray,
    tau: np.ndarray,
    noise: float,
    ground_threshold: int,
) -> tuple[float, float, float]:
    flags: list[bool] = []
    for steps in step_series_per_example:
        flags.append(should_abstain(steps, active, tau, noise, ground_threshold))
    prec, rec, f1, _, _, _ = abstention_f1(flags, labels)
    return prec, rec, f1


def run_grid(
    val_data: list[dict],
    save_csv: str | None,
    min_support_per_class: int,
) -> dict:
    chunk_sizes = list(range(50, 701, 50))
    deltas = [round(0.1 + 0.3 * i, 10) for i in range(34)]  # 0.1 .. 10.0 step 0.3
    noises = [round(1.0 + 0.5 * i, 10) for i in range(59)]  # 1.0 .. 30.0 step 0.5
    grounds = list(range(1, 51))

    thinking_ent = [token_entropies_thinking_only(d) for d in val_data]
    labels = [bool(d.get("is_correct")) for d in val_data]

    correct_cum_by_cs: dict[int, list[list[float]]] = {}
    incorrect_cum_by_cs: dict[int, list[list[float]]] = {}
    cum_flat_by_cs: dict[int, list[list[float]]] = {}

    for cs in chunk_sizes:
        csl, isl = [], []
        flat: list[list[float]] = []
        for d, ent in zip(val_data, thinking_ent):
            cum = cumulative_sum_chunk_means(chunk_step_means(ent, cs))
            flat.append(cum)
            if d.get("is_correct"):
                csl.append(cum)
            else:
                isl.append(cum)
        correct_cum_by_cs[cs] = csl
        incorrect_cum_by_cs[cs] = isl
        cum_flat_by_cs[cs] = flat

    best: dict | None = None
    rows: list[dict] = []

    for chunk_size in chunk_sizes:
        mean_corr, mean_inc, n_corr, n_inc = val_step_means_and_counts_from_step_lists(
            correct_cum_by_cs[chunk_size],
            incorrect_cum_by_cs[chunk_size],
        )
        for delta in deltas:
            active, tau = build_active_and_tau_with_min_support(
                mean_corr, mean_inc, delta, n_corr, n_inc, min_support_per_class
            )
            for noise in noises:
                for g in grounds:
                    prec, rec, f1 = evaluate_split_cached(
                        cum_flat_by_cs[chunk_size],
                        labels,
                        active,
                        tau,
                        noise,
                        g,
                    )
                    row = {
                        "chunk_size": chunk_size,
                        "delta": delta,
                        "noise": noise,
                        "ground_threshold": g,
                        "min_support_per_class": min_support_per_class,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                    }
                    rows.append(row)
                    if best is None or f1 > best["f1"] or (
                        f1 == best["f1"] and chunk_size < best["chunk_size"]
                    ):
                        best = dict(row)

    if save_csv and rows:
        with open(save_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    return best or {}


def _format_parameter_type(best: dict) -> str:
    return (
        f"chunk_size={int(best['chunk_size'])}, "
        f"delta={float(best['delta'])}, "
        f"noise={float(best['noise'])}, "
        f"ground_threshold={int(best['ground_threshold'])}, "
        f"min_support_per_class={int(best['min_support_per_class'])}"
    )


def run_abstention_evaluate(
    val_data: list[dict],
    test_data: list[dict],
    best: dict,
) -> dict:
    chunk_size = int(best["chunk_size"])
    delta = float(best["delta"])
    noise = float(best["noise"])
    ground_threshold = int(best["ground_threshold"])
    min_support = int(best.get("min_support_per_class", 3))

    mean_corr, mean_inc, n_corr, n_inc = val_step_cumulative_entropy_stats(
        val_data, chunk_size
    )
    active, tau = build_active_and_tau_with_min_support(
        mean_corr, mean_inc, delta, n_corr, n_inc, min_support
    )

    abstain_flags: list[bool] = []
    token_lens = [total_tokens_in_response(d) for d in test_data]
    saved_tokens = 0

    for d in test_data:
        cum = cumulative_sum_chunk_means(
            chunk_step_means(token_entropies_thinking_only(d), chunk_size)
        )
        ab = should_abstain(cum, active, tau, noise, ground_threshold)
        abstain_flags.append(ab)
        if ab:
            saved_tokens += token_lens[len(abstain_flags) - 1]

    labels = [bool(d.get("is_correct")) for d in test_data]
    _, _, _, tp, fp, _ = abstention_f1(abstain_flags, labels)

    n = len(test_data)
    n_test_correct = sum(1 for x in labels if x)
    n_test_incorrect = n - n_test_correct
    baseline_acc = n_test_correct / n if n else 0.0

    answered = [i for i, a in enumerate(abstain_flags) if not a]
    test_acc = sum(1 for i in answered if labels[i]) / n if n else 0.0

    n_abstain = sum(abstain_flags)
    mean_saved = (saved_tokens / n_abstain) if n_abstain else 0.0

    return {
        "val_precision": float(best["precision"]),
        "val_recall": float(best["recall"]),
        "val_f1": float(best["f1"]),
        "test_baseline_accuracy": baseline_acc,
        "test_n_correct": n_test_correct,
        "test_n_incorrect": n_test_incorrect,
        "test_accuracy": test_acc,
        "test_abstained_incorrect": tp,
        "test_abstained_correct": fp,
        "mean_tokens_saved_per_abstain": mean_saved,
        "saved_tokens_total": saved_tokens,
        "abstain_flags": abstain_flags,
        "mean_corr": mean_corr,
        "mean_inc": mean_inc,
        "active": active,
        "tau": tau,
        "chunk_size": chunk_size,
        "delta": delta,
        "noise": noise,
        "ground_threshold": ground_threshold,
        "min_support_per_class": min_support,
    }


def val_step_cumulative_entropy_stats(
    val_data: list[dict],
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validation mean cumulative entropy per step plus per-step class counts."""
    correct_steps: list[list[float]] = []
    incorrect_steps: list[list[float]] = []
    for d in val_data:
        ent = token_entropies_thinking_only(d)
        cum = cumulative_sum_chunk_means(chunk_step_means(ent, chunk_size))
        if d.get("is_correct"):
            correct_steps.append(cum)
        else:
            incorrect_steps.append(cum)
    return val_step_means_and_counts_from_step_lists(correct_steps, incorrect_steps)


def plot_validation_step_agg_entropy_curves(
    mean_corr: np.ndarray,
    mean_inc: np.ndarray,
    active: np.ndarray,
    tau: np.ndarray,
    noise: float,
    chunk_size: int,
    delta: float,
    out_path: str,
    model_name: str | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "Plotting requires matplotlib. Install with: pip install matplotlib"
        ) from e

    j = np.arange(len(mean_corr))
    fig, ax = plt.subplots(figsize=(12, 6))

    mask_c = ~np.isnan(mean_corr)
    mask_i = ~np.isnan(mean_inc)
    if np.any(mask_c):
        ax.plot(
            j[mask_c],
            mean_corr[mask_c],
            color="tab:green",
            linewidth=2,
            label="Val — mean cum. entropy (correct)",
        )
    if np.any(mask_i):
        ax.plot(
            j[mask_i],
            mean_inc[mask_i],
            color="tab:orange",
            linewidth=2,
            label="Val — mean cum. entropy (incorrect)",
        )

    act_idx = np.where(active)[0]
    if len(act_idx) > 0:
        ax.scatter(
            act_idx,
            tau[act_idx],
            color="tab:red",
            s=45,
            zorder=5,
            label="τ (active steps only)",
        )
        ax.scatter(
            act_idx,
            tau[act_idx] + noise,
            color="tab:purple",
            s=38,
            marker="x",
            zorder=5,
            label="τ + noise (abstain if cum. entropy >)",
        )

    ax.set_xlabel("Step index (non-overlapping chunk)")
    ax.set_ylabel("Cumulative chunk entropy (sum of mean H per chunk, 0…t)")
    title = (
        f"Validation: mean cumulative entropy vs step | chunk_size={chunk_size}, "
        f"δ={delta}, noise={noise}"
    )
    if model_name:
        title = f"{model_name}\n{title}"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_test_report(
    val_data: list[dict],
    test_data: list[dict],
    best: dict,
    plot_path: str | None = None,
    model_name: str | None = None,
) -> dict:
    m = run_abstention_evaluate(val_data, test_data, best)

    if plot_path:
        try:
            plot_validation_step_agg_entropy_curves(
                m["mean_corr"],
                m["mean_inc"],
                m["active"],
                m["tau"],
                m["noise"],
                m["chunk_size"],
                m["delta"],
                plot_path,
                model_name=model_name,
            )
            print(f"\nSaved validation cumulative-entropy abstention plot: {plot_path}")
        except ImportError as e:
            print(f"\nSkipping plot ({e})")

    chunk_size = m["chunk_size"]
    delta = m["delta"]
    noise = m["noise"]
    ground_threshold = m["ground_threshold"]
    min_sup = m["min_support_per_class"]
    labels = [bool(d.get("is_correct")) for d in test_data]
    prec_t, rec_t, f1_t, tp, fp, _ = abstention_f1(m["abstain_flags"], labels)

    n = len(test_data)
    n_test_correct = m["test_n_correct"]
    n_test_incorrect = m["test_n_incorrect"]
    baseline_acc = m["test_baseline_accuracy"]
    acc = m["test_accuracy"]
    abstain_flags = m["abstain_flags"]
    n_abstain = sum(abstain_flags)
    token_lens = [total_tokens_in_response(d) for d in test_data]
    saved_tokens = m["saved_tokens_total"]

    n_val = len(val_data)
    n_val_correct = sum(1 for d in val_data if d.get("is_correct"))
    n_val_incorrect = n_val - n_val_correct

    print("\n=== Validation set (counts) ===")
    print(f"  Correct: {n_val_correct}, incorrect: {n_val_incorrect}  (N={n_val})")

    print("\n=== Test set (τ/active from validation, cumulative entropy steps) ===")
    print(
        f"chunk_size={chunk_size}, delta={delta}, noise={noise}, "
        f"ground_threshold={ground_threshold}, min_support_per_class={min_sup}"
    )
    print(f"Baseline accuracy (all test examples, no abstention): {baseline_acc:.4f}")
    print(f"  Correct responses: {n_test_correct}, incorrect responses: {n_test_incorrect}  (N={n})")
    print(f"Test accuracy (non-abstained & correct) / N: {acc:.4f}  (N={n})")
    print(f"Abstentions: {n_abstain} / {n}")
    print(f"  Among abstained — incorrect (desired): {tp}")
    print(f"  Among abstained — correct (false abstain): {fp}")
    print(f"Abstention precision (test): {prec_t:.4f}, recall (test): {rec_t:.4f}, F1 (test): {f1_t:.4f}")
    print(f"Total tokens 'saved' (sum of full response lengths for abstained): {saved_tokens}")
    if n_abstain:
        print(f"Mean tokens saved per abstained response: {saved_tokens / n_abstain:.1f}")
    print(f"Mean full response length (all test): {float(np.mean(token_lens)):.1f}")

    return m


SUMMARY_CSV_COLUMNS = [
    "Model Name",
    "Type of Parameters",
    "Baseline Accuracy (test dataset)",
    "Number of Correct Responses (test dataset)",
    "Number of Incorrect Responses (test dataset)",
    "F1 Validation Score",
    "Precision on validation",
    "Recall on validation",
    "test accuracy",
    "among abstained incorrect (test dataset)",
    "Among abstained correct (test dataset)",
    "Mean token saved per abstain response (test dataset)",
]


def metrics_to_summary_row(model_name: str, best: dict, m: dict) -> dict[str, str | int | float]:
    return {
        "Model Name": model_name,
        "Type of Parameters": _format_parameter_type(best),
        "Baseline Accuracy (test dataset)": round(m["test_baseline_accuracy"], 6),
        "Number of Correct Responses (test dataset)": m["test_n_correct"],
        "Number of Incorrect Responses (test dataset)": m["test_n_incorrect"],
        "F1 Validation Score": round(m["val_f1"], 6),
        "Precision on validation": round(m["val_precision"], 6),
        "Recall on validation": round(m["val_recall"], 6),
        "test accuracy": round(m["test_accuracy"], 6),
        "among abstained incorrect (test dataset)": m["test_abstained_incorrect"],
        "Among abstained correct (test dataset)": m["test_abstained_correct"],
        "Mean token saved per abstain response (test dataset)": round(
            m["mean_tokens_saved_per_abstain"], 4
        ),
    }


def _sanitize_filename_component(name: str) -> str:
    safe = "".join(c if c not in r'\/:*?"<>|' else "_" for c in name)
    return safe.strip() or "model"


def run_dataset_batch(
    dataset_dir: str,
    model_names: list[str],
    abstaining_results_dir: str,
    abstaining_plots_dir: str,
    val_size: int,
    seed: int,
    min_support_per_class: int,
) -> None:
    """
    Batch mode: writes ``avg_entropy.csv`` only (summary; same as pre-refactor batch).
    No combined ``grid.csv`` (original behavior).
    """
    dataset_path = Path(dataset_dir)
    dataset_name = dataset_path.name
    model_dirs = discover_models_in_dataset(dataset_dir, model_names)
    if not model_dirs:
        print(f"No usable model directories under {dataset_dir}")
        return

    plots_dir = Path(abstaining_plots_dir) / dataset_name / METHOD_AGG_ENTROPY
    results_dir = Path(abstaining_results_dir) / dataset_name / METHOD_AGG_ENTROPY
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, str | int | float]] = []

    for model_name, res_dir in model_dirs:
        print(f"\n{'='*60}\nModel: {model_name}\n{res_dir}\n{'='*60}")
        data = load_results_flat_dir(res_dir)
        n_raw = len(data)
        data = filter_usable_examples_entropy(data)
        if len(data) != n_raw:
            print(
                f"Dropped {n_raw - len(data)} examples without usable tokens/top_logprobs; "
                f"{len(data)} remain."
            )
        n = len(data)
        if n <= val_size:
            print(
                f"SKIP: need more than val_size={val_size} examples, got {n}"
            )
            continue
        n_usable = len(data)
        val_data, test_data = stratified_val_test_split(data, val_size, seed)
        print(
            f"Stratified split on {n_usable} usable examples: "
            f"validation={len(val_data)}, test={len(test_data)} (val_size={val_size})"
        )

        best = run_grid(
            val_data, save_csv=None, min_support_per_class=min_support_per_class
        )
        if not best:
            print("SKIP: no grid results.")
            continue

        print("\n=== Best on validation (max F1) ===")
        for k, v in best.items():
            print(f"  {k}: {v}")

        safe = _sanitize_filename_component(model_name)
        plot_path = os.path.join(plots_dir, f"{safe}_val_step_agg_entropy.png")

        m = print_test_report(
            val_data,
            test_data,
            best,
            plot_path=plot_path,
            model_name=model_name,
        )
        summary_rows.append(metrics_to_summary_row(model_name, best, m))

    if not summary_rows:
        print("No models produced results; result CSVs not written.")
        return

    summary_path = results_dir / "avg_entropy.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)
    print(f"\nWrote summary CSV ({len(summary_rows)} models): {summary_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Cumulative step-entropy abstention grid + test (single dir or batch over outputs layout)"
    )
    p.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory with result_0.json, ... (single-model mode)",
    )
    p.add_argument(
        "--outputs_dir",
        type=str,
        default=None,
        help="Root with subdirs outputs/<dataset>/<model>/ (batch mode; use with --dataset)",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset folder name under --outputs_dir (batch mode)",
    )
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model folder names to include (batch mode). If omitted, all models under the dataset are used.",
    )
    p.add_argument(
        "--abstaining_results_dir",
        type=str,
        default="abstaining_results",
        help="Root for abstaining_results/<dataset>/agg_entropy/avg_entropy.csv (batch mode; summary only)",
    )
    p.add_argument(
        "--abstaining_plots_dir",
        type=str,
        default="abstaining_plots",
        help="Root for abstaining_plots/<dataset>/agg_entropy/*.png (batch mode)",
    )
    p.add_argument("--val_size", type=int, default=60, help="Validation set size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--min_support_per_class",
        type=int,
        default=3,
        help="Min correct and min incorrect val responses at a step for it to be active (default: 3)",
    )
    p.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Write full grid search CSV (single mode; can be very large)",
    )
    p.add_argument(
        "--plot_path",
        type=str,
        default=None,
        help="Validation plot (single mode). Default: <results_dir>/abstain_val_step_agg_entropy.png",
    )
    p.add_argument(
        "--no_plot",
        action="store_true",
        help="Do not write the validation plot.",
    )
    args = p.parse_args()

    if args.outputs_dir is not None:
        if not args.dataset:
            p.error("--outputs_dir requires --dataset")
        run_dataset_batch(
            os.path.join(args.outputs_dir, args.dataset),
            list(args.models or []),
            args.abstaining_results_dir,
            args.abstaining_plots_dir,
            args.val_size,
            args.seed,
            args.min_support_per_class,
        )
        return

    if not args.results_dir:
        p.error("Provide --results_dir (single model) or --outputs_dir and --dataset (batch)")

    data = load_results_flat_dir(args.results_dir)
    print(f"Loaded {len(data)} result files from {args.results_dir}")
    n_raw = len(data)
    data = filter_usable_examples_entropy(data)
    if len(data) != n_raw:
        print(
            f"Dropped {n_raw - len(data)} examples without usable tokens/top_logprobs; "
            f"{len(data)} remain."
        )

    n_usable = len(data)
    val_data, test_data = stratified_val_test_split(data, args.val_size, args.seed)
    print(
        f"Stratified split on {n_usable} usable examples: "
        f"validation={len(val_data)}, test={len(test_data)} (val_size={args.val_size})"
    )

    print(
        "Grid: chunk_size 50..700 step 50; delta 0.1..10 step 0.3; "
        "noise 1..30 step 0.5; ground_threshold 1..50. This may take a long time."
    )
    best = run_grid(val_data, args.output_csv, args.min_support_per_class)
    if not best:
        print("No grid results.")
        return

    print("\n=== Best on validation (max F1) ===")
    for k, v in best.items():
        print(f"  {k}: {v}")

    plot_path: str | None = None
    if not args.no_plot:
        plot_path = args.plot_path or os.path.join(
            args.results_dir, "abstain_val_step_agg_entropy.png"
        )

    print_test_report(
        val_data,
        test_data,
        best,
        plot_path=plot_path,
        model_name=None,
    )


if __name__ == "__main__":
    main()
