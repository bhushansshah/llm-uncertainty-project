#!/usr/bin/env python3
"""
Step-entropy abstention experiment (validation grid search + test evaluation).

Hyperparameter grids are **data-driven** from the validation split (see ``run_grid``):

  - **chunk_size:** 50, 100, … up to the 70th percentile of thinking-token lengths (step 50).
  - **delta:** 20 values from μ − 2.5σ to μ + 2.5σ where μ, σ are the mean and std of
    (incorrect_mean − correct_mean) over steps with incorrect > correct and min support.
  - **offset** (formerly noise): for each (chunk_size, δ), up to 20 values from 0 to the 95th
    percentile of 2.5 × within-step std of correct responses on active steps.
  - **ground_threshold:** 1..20 (min active steps exceeding τ + offset to abstain).

Metrics:
  - Validation: maximize F1 of abstention (TP = abstain & wrong, FP = abstain & correct)
  - ``grid.csv`` rows include ``validation_accuracy``: (# non-abstained & correct on validation) /
    N_val for each hyperparameter row (same abstention policy as precision/recall/F1).
  - Test accuracy: (# non-abstained & is_correct) / N_test (abstain counts as failure)

Active steps additionally require at least ``min_support_per_class`` correct **and** incorrect
validation responses at that step (default: 3), same as ``abstain_step_neg_logprob_experiment``.
"""

from __future__ import annotations

import argparse
import csv
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
from abstain_experiment_common import (  # noqa: E402
    load_results_flat_dir,
    run_step_signal_grid,
    sanitize_filename_component,
    stratified_val_test_split_fraction as stratified_val_test_split,
)
from abstain_step_entropy import (  # noqa: E402
    abstention_f1,
    build_active_and_tau_with_min_support,
    chunk_step_means,
    filter_usable_examples_entropy,
    should_abstain,
    token_entropies_thinking_only,
    total_tokens_in_response,
    val_step_means_and_counts_from_step_lists,
)


def run_grid(
    val_data: list[dict],
    save_csv: str | None,
    min_support_per_class: int,
) -> tuple[dict, list, list[tuple[int, float, np.ndarray]]]:
    thinking_ent = [token_entropies_thinking_only(d) for d in val_data]
    return run_step_signal_grid(
        val_data, thinking_ent, save_csv, min_support_per_class
    )


def _format_parameter_type(best: dict) -> str:
    """Human-readable best hyperparameters for CSV."""
    off = best.get("offset", best.get("noise"))
    return (
        f"chunk_size={int(best['chunk_size'])}, "
        f"delta={float(best['delta'])}, "
        f"offset={float(off)}, "
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
    Test abstention precision/recall/F1 are computed on the test split with those hyperparameters.
    """
    chunk_size = int(best["chunk_size"])
    delta = float(best["delta"])
    offset = float(best.get("offset", best.get("noise", 0.0)))
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
        ab = should_abstain(steps, active, tau, offset, ground_threshold)
        abstain_flags.append(ab)
        if ab:
            saved_tokens += token_lens[len(abstain_flags) - 1]

    labels = [bool(d.get("is_correct")) for d in test_data]
    prec_t, rec_t, f1_t, tp, fp, _ = abstention_f1(abstain_flags, labels)

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
        "test_abstention_precision": float(prec_t),
        "test_abstention_recall": float(rec_t),
        "test_abstention_f1": float(f1_t),
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
        "offset": offset,
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
    offset: float,
    chunk_size: int,
    delta: float,
    out_path: str,
    model_name: str | None = None,
) -> None:
    """
    Like test.ipynb aggregate plots: mean chunk entropy vs step index on validation,
    plus τ at active steps and τ + offset (abstention comparison level).
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
            tau[act_idx] + offset,
            color="tab:purple",
            s=38,
            marker="x",
            zorder=5,
            label="τ + offset (abstain if step mean >)",
        )

    ax.set_xlabel("Step index (non-overlapping chunk)")
    ax.set_ylabel("Mean chunk entropy (thinking tokens)")
    title = f"Validation: mean entropy vs step | chunk_size={chunk_size}, δ={delta}, offset={offset}"
    if model_name:
        title = f"{model_name}\n{title}"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_marks(
    mean_corr: np.ndarray,
    mean_inc: np.ndarray,
    tau_records: list[tuple[int, float, np.ndarray]],
    out_path: str,
    model_name: str | None = None,
) -> None:
    """
    Validation mean chunk entropy (correct vs incorrect) vs step, plus every τ[j] from the
    grid (finite values) so all per-step thresholds tried in the experiment are visible.
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

    xs: list[int] = []
    ys: list[float] = []
    for _cs, _d, tau in tau_records:
        for step in range(len(tau)):
            v = tau[step]
            if np.isfinite(v):
                xs.append(step)
                ys.append(float(v))
    if xs:
        ax.scatter(
            xs,
            ys,
            s=10,
            alpha=0.35,
            c="tab:red",
            edgecolors="none",
            label="τ (all grid chunk_size, δ)",
            rasterized=True,
        )

    ax.set_xlabel("Step index (non-overlapping chunk)")
    ax.set_ylabel("Mean chunk entropy / τ (thinking tokens)")
    title = "Validation: mean entropy vs step with all τ from grid search"
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
    tau_records: list[tuple[int, float, np.ndarray]] | None = None,
    threshold_marks_path: str | None = None,
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
                m["offset"],
                m["chunk_size"],
                m["delta"],
                plot_path,
                model_name=model_name,
            )
            print(f"\nSaved validation step-entropy plot: {plot_path}")
        except ImportError as e:
            print(f"\nSkipping plot ({e})")

    if threshold_marks_path and tau_records:
        try:
            os.makedirs(os.path.dirname(threshold_marks_path) or ".", exist_ok=True)
            plot_threshold_marks(
                m["mean_corr"],
                m["mean_inc"],
                tau_records,
                threshold_marks_path,
                model_name=model_name,
            )
            print(f"\nSaved threshold-marks plot: {threshold_marks_path}")
        except ImportError as e:
            print(f"\nSkipping threshold-marks plot ({e})")

    chunk_size = m["chunk_size"]
    delta = m["delta"]
    offset = m["offset"]
    ground_threshold = m["ground_threshold"]
    min_sup = m["min_support_per_class"]
    labels = [bool(d.get("is_correct")) for d in test_data]
    prec_t = m["test_abstention_precision"]
    rec_t = m["test_abstention_recall"]
    f1_t = m["test_abstention_f1"]
    tp = m["test_abstained_incorrect"]
    fp = m["test_abstained_correct"]

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
        f"chunk_size={chunk_size}, delta={delta}, offset={offset}, "
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
    "test accuracy",
    "Number of Correct Responses (test dataset)",
    "Number of Incorrect Responses (test dataset)",
    "F1 Validation Score",
    "F1 on test (abstention)",
    "Precision on validation",
    "Precision on test (abstention)",
    "Recall on validation",
    "Recall on test (abstention)",
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
        "test accuracy": round(m["test_accuracy"], 6),
        "Number of Correct Responses (test dataset)": m["test_n_correct"],
        "Number of Incorrect Responses (test dataset)": m["test_n_incorrect"],
        "F1 Validation Score": round(m["val_f1"], 6),
        "F1 on test (abstention)": round(m["test_abstention_f1"], 6),
        "Precision on validation": round(m["val_precision"], 6),
        "Precision on test (abstention)": round(m["test_abstention_precision"], 6),
        "Recall on validation": round(m["val_recall"], 6),
        "Recall on test (abstention)": round(m["test_abstention_recall"], 6),
        "among abstained incorrect (test dataset)": m["test_abstained_incorrect"],
        "Among abstained correct (test dataset)": m["test_abstained_correct"],
        "Mean token saved per abstain response (test dataset)": round(
            m["mean_tokens_saved_per_abstain"], 4
        ),
    }


def run_dataset_batch(
    dataset_dir: str,
    model_names: list[str],
    abstaining_results_dir: str,
    abstaining_plots_dir: str,
    abstaining_threshold_marks_dir: str,
    val_fraction: float,
    seed: int,
    min_support_per_class: int,
) -> None:
    """
    ``dataset_dir`` is ``<outputs_root>/<dataset_name>/``. Writes ``avg_entropy.csv`` and
    ``grid.csv`` under ``abstaining_results_dir/<dataset>/step_entropy/``, plots under
    ``abstaining_plots_dir/<dataset>/step_entropy/``, threshold-marks plots under
    ``abstaining_threshold_marks_dir/<dataset>/step_entropy/``.
    """
    dataset_path = Path(dataset_dir)
    dataset_name = dataset_path.name
    model_dirs = discover_models_in_dataset(dataset_dir, model_names)
    if not model_dirs:
        print(f"No usable model directories under {dataset_dir}")
        return

    plots_dir = Path(abstaining_plots_dir) / dataset_name / METHOD_STEP_ENTROPY
    marks_dir = Path(abstaining_threshold_marks_dir) / dataset_name / METHOD_STEP_ENTROPY
    results_dir = Path(abstaining_results_dir) / dataset_name / METHOD_STEP_ENTROPY
    plots_dir.mkdir(parents=True, exist_ok=True)
    marks_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, str | int | float]] = []
    combined_grid: list[dict[str, str | int | float]] = []

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
        if n < 2:
            print(f"SKIP: need at least 2 usable examples, got {n}")
            continue
        try:
            val_data, test_data = stratified_val_test_split(data, val_fraction, seed)
        except ValueError as e:
            print(f"SKIP: {e}")
            continue
        n_usable = len(data)
        print(
            f"Stratified split on {n_usable} usable examples: "
            f"validation={len(val_data)}, test={len(test_data)} "
            f"(val_fraction={val_fraction})"
        )

        best, grid_rows, tau_records = run_grid(
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

        safe = sanitize_filename_component(model_name)
        plot_path = os.path.join(plots_dir, f"{safe}_val_step_entropy.png")
        threshold_marks_path = os.path.join(marks_dir, f"{safe}_threshold_marks.png")

        m = print_test_report(
            val_data,
            test_data,
            best,
            plot_path=plot_path,
            model_name=model_name,
            tau_records=tau_records,
            threshold_marks_path=threshold_marks_path,
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
    p.add_argument(
        "--abstaining_threshold_marks_dir",
        type=str,
        default="abstaining_threshold_marks",
        help="Root for abstaining_threshold_marks/<dataset>/step_entropy/*_threshold_marks.png (batch/single)",
    )
    p.add_argument(
        "--val_fraction",
        type=float,
        default=0.3,
        help="Fraction of usable responses in the validation set (e.g. 0.3 → 30%%); rest is test",
    )
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
    if not (0.0 < args.val_fraction < 1.0):
        p.error("--val_fraction must be strictly between 0 and 1 (e.g. 0.3 for 30%% validation)")

    if args.outputs_dir is not None:
        if not args.dataset:
            p.error("--outputs_dir requires --dataset")
        run_dataset_batch(
            os.path.join(args.outputs_dir, args.dataset),
            list(args.models or []),
            args.abstaining_results_dir,
            args.abstaining_plots_dir,
            args.abstaining_threshold_marks_dir,
            args.val_fraction,
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
    if n_usable < 2:
        raise SystemExit(f"Need at least 2 usable examples after filtering, got {n_usable}")
    val_data, test_data = stratified_val_test_split(data, args.val_fraction, args.seed)
    print(
        f"Stratified split on {n_usable} usable examples: "
        f"validation={len(val_data)}, test={len(test_data)} "
        f"(val_fraction={args.val_fraction})"
    )

    best, _, tau_records = run_grid(
        val_data, args.output_csv, args.min_support_per_class
    )
    if not best:
        print("No grid results.")
        return

    print("\n=== Best on validation (max F1) ===")
    for k, v in best.items():
        print(f"  {k}: {v}")

    plot_path: str | None = None
    threshold_marks_path: str | None = None
    if not args.no_plot:
        plot_path = args.plot_path or os.path.join(
            args.results_dir, "abstain_val_step_entropy.png"
        )
        res_name = Path(args.results_dir).name
        tm_dir = Path(args.abstaining_threshold_marks_dir) / res_name / METHOD_STEP_ENTROPY
        tm_dir.mkdir(parents=True, exist_ok=True)
        safe = sanitize_filename_component(res_name)
        threshold_marks_path = str(tm_dir / f"{safe}_threshold_marks.png")

    print_test_report(
        val_data,
        test_data,
        best,
        plot_path=plot_path,
        model_name=None,
        tau_records=tau_records,
        threshold_marks_path=threshold_marks_path,
    )


if __name__ == "__main__":
    main()
