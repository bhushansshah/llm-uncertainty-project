#!/usr/bin/env python3
"""
Plot **cumulative positive increments** between consecutive chunk mean entropies vs step (validation only).

For each response, non-overlapping **chunk** means of thinking-token entropies are computed
(``chunk_size`` tokens per chunk). At step ``j``, the value is::

    sum_{i=1}^{j} max(0, m[i] - m[i-1])

with ``m[i]`` the mean entropy in chunk ``i``, and step ``0`` defined as ``0``. Same idea as
``test.ipynb`` (positive variation / cumulative positive differences between consecutive steps).

Layout matches ``val_step_agg_entropy_plot.py``: three stacked panels (mean, variance, counts),
``sharex=True``. Optional ``--separate_plots``.

Usage:
  python scripts/val_step_pos_entropy_diff_plot.py \\
    --results_dir path/to/result_jsons \\
    --val_size 60 \\
    --chunk_size 50 \\
    --output_path val_pos_entropy_diff_vs_step.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from abstain_step_entropy import (  # noqa: E402
    chunk_step_means,
    cumulative_positive_chunk_increments,
    token_entropies_thinking_only,
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
    return [data[i] for i in idx[:val_size]], [data[i] for i in idx[val_size:]]


def val_step_mean_var_count_from_step_lists(
    correct_steps: list[list[float]],
    incorrect_steps: list[list[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_len = 0
    for s in correct_steps + incorrect_steps:
        max_len = max(max_len, len(s))

    mean_corr = np.full(max_len, np.nan, dtype=float)
    mean_inc = np.full(max_len, np.nan, dtype=float)
    var_corr = np.full(max_len, np.nan, dtype=float)
    var_inc = np.full(max_len, np.nan, dtype=float)
    n_corr = np.zeros(max_len, dtype=int)
    n_inc = np.zeros(max_len, dtype=int)

    for j in range(max_len):
        cvals = [s[j] for s in correct_steps if j < len(s)]
        ivals = [s[j] for s in incorrect_steps if j < len(s)]
        n_corr[j] = len(cvals)
        n_inc[j] = len(ivals)
        if cvals:
            mean_corr[j] = float(np.mean(cvals))
            var_corr[j] = float(np.var(cvals, ddof=0))
        if ivals:
            mean_inc[j] = float(np.mean(ivals))
            var_inc[j] = float(np.var(ivals, ddof=0))

    return mean_corr, mean_inc, var_corr, var_inc, n_corr, n_inc


def plot_val_mean_pos_diff_vs_step(
    mean_correct: np.ndarray,
    mean_incorrect: np.ndarray,
    chunk_size: int,
    out_path: str,
    title_suffix: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    j = np.arange(len(mean_correct))
    fig, ax = plt.subplots(figsize=(12, 6))

    mask_c = ~np.isnan(mean_correct)
    mask_i = ~np.isnan(mean_incorrect)
    if np.any(mask_c):
        ax.plot(
            j[mask_c],
            mean_correct[mask_c],
            color="tab:green",
            linewidth=2,
            label="Val — mean cum. pos. Δ (correct)",
        )
    if np.any(mask_i):
        ax.plot(
            j[mask_i],
            mean_incorrect[mask_i],
            color="tab:orange",
            linewidth=2,
            label="Val — mean cum. pos. Δ (incorrect)",
        )

    ax.set_xlabel("Step index (non-overlapping chunk)")
    ax.set_ylabel("Cumulative Σ max(0, Δ mean H between consecutive chunks)")
    title = (
        f"Validation: mean cumulative positive chunk jumps vs step | chunk_size={chunk_size}\n"
        "(per-step mean over val responses that reach that step)"
    )
    if title_suffix:
        title = f"{title_suffix}\n{title}"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_val_variance_pos_diff_vs_step(
    var_correct: np.ndarray,
    var_incorrect: np.ndarray,
    chunk_size: int,
    out_path: str,
    title_suffix: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    j = np.arange(len(var_correct))
    fig, ax = plt.subplots(figsize=(12, 6))

    mask_c = ~np.isnan(var_correct)
    mask_i = ~np.isnan(var_incorrect)
    if np.any(mask_c):
        ax.plot(
            j[mask_c],
            var_correct[mask_c],
            color="tab:green",
            linewidth=2,
            label="Val — Var(cum. pos. Δ) (correct)",
        )
    if np.any(mask_i):
        ax.plot(
            j[mask_i],
            var_incorrect[mask_i],
            color="tab:orange",
            linewidth=2,
            label="Val — Var(cum. pos. Δ) (incorrect)",
        )

    ax.set_xlabel("Step index (non-overlapping chunk)")
    ax.set_ylabel("Variance of cumulative positive chunk jumps")
    title = (
        f"Validation: variance of cumulative positive jumps vs step | chunk_size={chunk_size}\n"
        "(population variance across validation examples at each step)"
    )
    if title_suffix:
        title = f"{title_suffix}\n{title}"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_val_support_counts_vs_step(
    n_correct: np.ndarray,
    n_incorrect: np.ndarray,
    chunk_size: int,
    out_path: str,
    title_suffix: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    j = np.arange(len(n_correct))
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(j, n_correct, color="tab:green", linewidth=2, label="# val responses (correct)")
    ax.plot(j, n_incorrect, color="tab:orange", linewidth=2, label="# val responses (incorrect)")
    ax.set_xlabel("Step index (non-overlapping chunk)")
    ax.set_ylabel("Number of responses reaching this step")
    title = (
        f"Validation: support count vs step | chunk_size={chunk_size}\n"
        "(how many correct / incorrect examples have at least this many steps)"
    )
    if title_suffix:
        title = f"{title_suffix}\n{title}"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_val_pos_diff_combined(
    mean_correct: np.ndarray,
    mean_incorrect: np.ndarray,
    var_correct: np.ndarray,
    var_incorrect: np.ndarray,
    n_correct: np.ndarray,
    n_incorrect: np.ndarray,
    chunk_size: int,
    out_path: str,
    title_suffix: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    j = np.arange(len(mean_correct))
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 14),
        sharex=True,
        constrained_layout=True,
    )
    ax0, ax1, ax2 = axes

    mask_c = ~np.isnan(mean_correct)
    mask_i = ~np.isnan(mean_incorrect)
    if np.any(mask_c):
        ax0.plot(
            j[mask_c],
            mean_correct[mask_c],
            color="tab:green",
            linewidth=2,
            label="Mean cum. pos. Δ (correct)",
        )
    if np.any(mask_i):
        ax0.plot(
            j[mask_i],
            mean_incorrect[mask_i],
            color="tab:orange",
            linewidth=2,
            label="Mean cum. pos. Δ (incorrect)",
        )
    ax0.set_ylabel(
        "Cumulative Σ max(0, ΔH)\n(consecutive chunk means)"
    )
    ax0.set_title(
        "(1) Mean cumulative positive entropy jumps vs step — validation; "
        "per-step mean over responses that reach that step"
    )
    ax0.legend(loc="best", fontsize=8)
    ax0.grid(True, alpha=0.3)

    mask_vc = ~np.isnan(var_correct)
    mask_vi = ~np.isnan(var_incorrect)
    if np.any(mask_vc):
        ax1.plot(
            j[mask_vc],
            var_correct[mask_vc],
            color="tab:green",
            linewidth=2,
            label="Var(cum. pos. Δ) (correct)",
        )
    if np.any(mask_vi):
        ax1.plot(
            j[mask_vi],
            var_incorrect[mask_vi],
            color="tab:orange",
            linewidth=2,
            label="Var(cum. pos. Δ) (incorrect)",
        )
    ax1.set_ylabel("Variance of cumulative positive jumps")
    ax1.set_title("(2) Population variance across validation examples at each step")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(j, n_correct, color="tab:green", linewidth=2, label="# responses (correct)")
    ax2.plot(j, n_incorrect, color="tab:orange", linewidth=2, label="# responses (incorrect)")
    ax2.set_ylabel("Count")
    ax2.set_xlabel("Step index (non-overlapping chunk)")
    ax2.set_title("(3) Number of validation responses that reach each step")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    if len(j) > 0:
        ax2.set_xlim(j[0] - 0.5, j[-1] + 0.5)

    supt = (
        f"Validation — cumulative positive Δ between chunk mean entropies | "
        f"chunk_size={chunk_size} tokens/chunk"
    )
    if title_suffix:
        supt = f"{title_suffix}\n{supt}"
    fig.suptitle(supt, fontsize=12, y=1.02)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Plot validation mean cumulative positive entropy increments vs step "
            "(correct vs incorrect)"
        )
    )
    p.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory with result_0.json, ... (needs logprobs.tokens and top_logprobs)",
    )
    p.add_argument(
        "--val_size",
        type=int,
        default=60,
        help="Validation set size (only these examples are plotted)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--chunk_size",
        type=int,
        default=50,
        help="Tokens per chunk (same as abstention experiments)",
    )
    p.add_argument(
        "--output_path",
        type=str,
        default="val_pos_entropy_diff_vs_step.png",
        help="Output PNG for the combined 3-panel figure",
    )
    p.add_argument(
        "--separate_plots",
        action="store_true",
        help="Also save three standalone PNGs next to output stem",
    )
    p.add_argument(
        "--variance_output_path",
        type=str,
        default=None,
        help="With --separate_plots: variance PNG (default: <stem>_variance.png)",
    )
    p.add_argument(
        "--counts_output_path",
        type=str,
        default=None,
        help="With --separate_plots: counts PNG (default: <stem>_counts.png)",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional extra title line (e.g. model name)",
    )
    args = p.parse_args()

    data = load_results_flat_dir(args.results_dir)
    print(f"Loaded {len(data)} files from {args.results_dir}")

    val_data, _test_discard = stratified_val_test_split(data, args.val_size, args.seed)
    print(f"Using validation only: N_val={len(val_data)} (test split discarded for this plot)")

    correct_steps: list[list[float]] = []
    incorrect_steps: list[list[float]] = []
    for d in val_data:
        ent = token_entropies_thinking_only(d)
        chunk_means = chunk_step_means(ent, args.chunk_size)
        series = cumulative_positive_chunk_increments(chunk_means)
        if d.get("is_correct"):
            correct_steps.append(series)
        else:
            incorrect_steps.append(series)

    mean_corr, mean_inc, var_corr, var_inc, n_corr, n_inc = (
        val_step_mean_var_count_from_step_lists(correct_steps, incorrect_steps)
    )

    out_base = Path(args.output_path)
    var_path = (
        Path(args.variance_output_path)
        if args.variance_output_path
        else out_base.parent / f"{out_base.stem}_variance{out_base.suffix}"
    )
    cnt_path = (
        Path(args.counts_output_path)
        if args.counts_output_path
        else out_base.parent / f"{out_base.stem}_counts{out_base.suffix}"
    )

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    plot_val_pos_diff_combined(
        mean_corr,
        mean_inc,
        var_corr,
        var_inc,
        n_corr,
        n_inc,
        args.chunk_size,
        args.output_path,
        title_suffix=args.title,
    )
    print(f"Saved combined plot (3 panels, shared x-axis): {args.output_path}")

    if args.separate_plots:
        mean_only = out_base.parent / f"{out_base.stem}_mean_only{out_base.suffix}"
        plot_val_mean_pos_diff_vs_step(
            mean_corr,
            mean_inc,
            args.chunk_size,
            str(mean_only),
            title_suffix=args.title,
        )
        plot_val_variance_pos_diff_vs_step(
            var_corr,
            var_inc,
            args.chunk_size,
            str(var_path),
            title_suffix=args.title,
        )
        plot_val_support_counts_vs_step(
            n_corr,
            n_inc,
            args.chunk_size,
            str(cnt_path),
            title_suffix=args.title,
        )
        print(f"Also saved separate: {mean_only}, {var_path}, {cnt_path}")


if __name__ == "__main__":
    main()
