#!/usr/bin/env python3
"""
Step-entropy abstention experiment (validation grid search + test evaluation).

Hyperparameters (grid):
  - chunk_size: 50..600 step 50
  - delta: min gap between incorrect/correct mean step entropy on validation (0.02..0.08 step 0.01)
  - noise: added to tau when comparing step entropy (0.01..0.10 step 0.01)
  - ground_threshold: min number of "bad" active steps to abstain (1..10)

Metrics:
  - Validation: maximize F1 of abstention (TP = abstain & wrong, FP = abstain & correct)
  - Test accuracy: (# non-abstained & is_correct) / N_test (abstain counts as failure)

Active steps additionally require at least ``min_support_per_class`` correct **and** incorrect
validation responses at that step (default: 3), same as ``abstain_step_neg_logprob_experiment``.
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

# project root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from abstain_batch_utils import (  # noqa: E402
    METHOD_STEP_ENTROPY,
    discover_models_in_dataset,
)
from abstain_step_entropy import (  # noqa: E402
    abstention_f1,
    build_active_and_tau_with_min_support,
    chunk_step_means,
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

    # Stratify when possible (both classes present and enough samples per class)
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
    step_means_per_example: list[list[float]],
    labels: list[bool],
    active: np.ndarray,
    tau: np.ndarray,
    noise: float,
    ground_threshold: int,
) -> tuple[float, float, float]:
    flags: list[bool] = []
    for steps in step_means_per_example:
        flags.append(should_abstain(steps, active, tau, noise, ground_threshold))
    prec, rec, f1, _, _, _ = abstention_f1(flags, labels)
    return prec, rec, f1


def run_grid(
    val_data: list[dict],
    save_csv: str | None,
    min_support_per_class: int,
) -> tuple[dict, list]:
    chunk_sizes = list(range(50, 601, 50))
    deltas = [round(0.02 + 0.01 * i, 2) for i in range(7)]  # 0.02..0.08
    noises = [round(0.01 + 0.01 * i, 2) for i in range(10)]  # 0.01..0.10
    grounds = list(range(1, 11))

    # One thinking-entropy pass per validation example; then chunk once per chunk_size
    thinking_ent = [token_entropies_thinking_only(d) for d in val_data]
    labels = [bool(d.get("is_correct")) for d in val_data]
    correct_steps_by_cs: dict[int, list[list[float]]] = {}
    incorrect_steps_by_cs: dict[int, list[list[float]]] = {}
    for cs in chunk_sizes:
        csl, isl = [], []
        for d, ent in zip(val_data, thinking_ent):
            steps = chunk_step_means(ent, cs)
            if d.get("is_correct"):
                csl.append(steps)
            else:
                isl.append(steps)
        correct_steps_by_cs[cs] = csl
        incorrect_steps_by_cs[cs] = isl

    step_means_flat_by_cs: dict[int, list[list[float]]] = {}
    for cs in chunk_sizes:
        step_means_flat_by_cs[cs] = [
            chunk_step_means(ent, cs) for ent in thinking_ent
        ]

    best: dict | None = None
    rows = []

    for chunk_size in chunk_sizes:
        mean_corr, mean_inc, n_corr, n_inc = val_step_means_and_counts_from_step_lists(
            correct_steps_by_cs[chunk_size],
            incorrect_steps_by_cs[chunk_size],
        )
        for delta in deltas:
            active, tau = build_active_and_tau_with_min_support(
                mean_corr, mean_inc, delta, n_corr, n_inc, min_support_per_class
            )
            for noise in noises:
                for g in grounds:
                    prec, rec, f1 = evaluate_split_cached(
                        step_means_flat_by_cs[chunk_size],
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

    return (best or {}), rows


def _format_parameter_type(best: dict) -> str:
    """Human-readable best hyperparameters for CSV."""
    return (
        f"chunk_size={int(best['chunk_size'])}, "
        f"delta={float(best['delta'])}, "
        f"noise={float(best['noise'])}, "
        f"ground_threshold={int(best['ground_threshold'])}, "
        f"min_support_per_class={int(best.get('min_support_per_class', 3))}"
    )


def run_abstention_evaluate(
    val_data: list[dict],
    test_data: list[dict],
    best: dict,
) -> dict:
    """
    Run abstention on test using τ/active from validation best hyperparameters.
    Returns metrics; validation precision/recall/F1 come from `best` (grid search on val).
    """
    chunk_size = int(best["chunk_size"])
    delta = float(best["delta"])
    noise = float(best["noise"])
    ground_threshold = int(best["ground_threshold"])
    min_support = int(best.get("min_support_per_class", 3))

    mean_corr, mean_inc, n_corr, n_inc = val_step_entropy_stats(val_data, chunk_size)
    active, tau = build_active_and_tau_with_min_support(
        mean_corr, mean_inc, delta, n_corr, n_inc, min_support
    )

    abstain_flags: list[bool] = []
    token_lens = [total_tokens_in_response(d) for d in test_data]
    saved_tokens = 0

    for d in test_data:
        steps = chunk_step_means(token_entropies_thinking_only(d), chunk_size)
        ab = should_abstain(steps, active, tau, noise, ground_threshold)
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


def val_step_entropy_stats(
    val_data: list[dict],
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validation mean chunk entropy per step plus per-step class counts."""
    correct_steps: list[list[float]] = []
    incorrect_steps: list[list[float]] = []
    for d in val_data:
        steps = chunk_step_means(token_entropies_thinking_only(d), chunk_size)
        if d.get("is_correct"):
            correct_steps.append(steps)
        else:
            incorrect_steps.append(steps)
    return val_step_means_and_counts_from_step_lists(correct_steps, incorrect_steps)


def plot_validation_step_entropy_curves(
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
    """
    Like test.ipynb aggregate plots: mean chunk entropy vs step index on validation,
    plus τ at active steps and τ + noise (abstention comparison level).
    """
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
            label="Val — mean entropy (correct)",
        )
    if np.any(mask_i):
        ax.plot(
            j[mask_i],
            mean_inc[mask_i],
            color="tab:orange",
            linewidth=2,
            label="Val — mean entropy (incorrect)",
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
            label="τ + noise (abstain if step mean >)",
        )

    ax.set_xlabel("Step index (non-overlapping chunk)")
    ax.set_ylabel("Mean chunk entropy (thinking tokens)")
    title = f"Validation: mean entropy vs step | chunk_size={chunk_size}, δ={delta}, noise={noise}"
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
    """Run evaluation, optional plot, print summary; returns metrics dict from run_abstention_evaluate."""
    m = run_abstention_evaluate(val_data, test_data, best)

    if plot_path:
        try:
            plot_validation_step_entropy_curves(
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
            print(f"\nSaved validation step-entropy plot: {plot_path}")
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

    print("\n=== Test set (tau/active from validation) ===")
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
    """One row for the GPQA summary CSV."""
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
    ``dataset_dir`` is ``<outputs_root>/<dataset_name>/``. Writes ``avg_entropy.csv`` and
    ``grid.csv`` under ``abstaining_results_dir/<dataset>/step_entropy/``, plots under
    ``abstaining_plots_dir/<dataset>/step_entropy/``.
    """
    dataset_path = Path(dataset_dir)
    dataset_name = dataset_path.name
    model_dirs = discover_models_in_dataset(dataset_dir, model_names)
    if not model_dirs:
        print(f"No usable model directories under {dataset_dir}")
        return

    plots_dir = Path(abstaining_plots_dir) / dataset_name / METHOD_STEP_ENTROPY
    results_dir = Path(abstaining_results_dir) / dataset_name / METHOD_STEP_ENTROPY
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, str | int | float]] = []
    combined_grid: list[dict[str, str | int | float]] = []

    for model_name, res_dir in model_dirs:
        print(f"\n{'='*60}\nModel: {model_name}\n{res_dir}\n{'='*60}")
        data = load_results_flat_dir(res_dir)
        n = len(data)
        if n <= val_size:
            print(
                f"SKIP: need more than val_size={val_size} examples, got {n}"
            )
            continue
        val_data, test_data = stratified_val_test_split(data, val_size, seed)
        print(f"Validation: {len(val_data)}, Test: {len(test_data)}")

        best, grid_rows = run_grid(
            val_data, save_csv=None, min_support_per_class=min_support_per_class
        )
        if not best:
            print("SKIP: no grid results.")
            continue

        for r in grid_rows:
            combined_grid.append({"model_name": model_name, **r})

        print("\n=== Best on validation (max F1) ===")
        for k, v in best.items():
            print(f"  {k}: {v}")

        safe = _sanitize_filename_component(model_name)
        plot_path = os.path.join(plots_dir, f"{safe}_val_step_entropy.png")

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

    if combined_grid:
        grid_path = results_dir / "grid.csv"
        grid_fields = ["model_name"] + [
            k for k in combined_grid[0].keys() if k != "model_name"
        ]
        with open(grid_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=grid_fields)
            w.writeheader()
            w.writerows(combined_grid)
        print(
            f"Wrote combined validation grid ({len(combined_grid)} rows): {grid_path}"
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Step-entropy abstention grid search + test eval (single dir or batch over outputs layout)"
    )
    p.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory containing result_0.json, ... (single-model mode)",
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
        help="Root for abstaining_results/<dataset>/step_entropy/avg_entropy.csv and grid.csv (batch mode)",
    )
    p.add_argument(
        "--abstaining_plots_dir",
        type=str,
        default="abstaining_plots",
        help="Root for abstaining_plots/<dataset>/step_entropy/*.png (batch mode)",
    )
    p.add_argument("--val_size", type=int, default=60, help="Validation set size (rest is test)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--min_support_per_class",
        type=int,
        default=3,
        help="Min correct and min incorrect val responses at a step for it to be active (default: 3)",
    )
    p.add_argument("--output_csv", type=str, default=None, help="Write full grid results to CSV (single mode)")
    p.add_argument(
        "--plot_path",
        type=str,
        default=None,
        help="Save validation plot (single mode). Default: <results_dir>/abstain_val_step_entropy.png",
    )
    p.add_argument(
        "--no_plot",
        action="store_true",
        help="Do not write the validation step-entropy plot.",
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

    val_data, test_data = stratified_val_test_split(data, args.val_size, args.seed)
    print(f"Validation: {len(val_data)}, Test: {len(test_data)}")

    best, _ = run_grid(val_data, args.output_csv, args.min_support_per_class)
    if not best:
        print("No grid results.")
        return

    print("\n=== Best on validation (max F1) ===")
    for k, v in best.items():
        print(f"  {k}: {v}")

    plot_path: str | None = None
    if not args.no_plot:
        plot_path = args.plot_path or os.path.join(
            args.results_dir, "abstain_val_step_entropy.png"
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
