#!/usr/bin/env python3
"""
Plot **per-step mean chunk KL(U‖p)** on the **validation** split only.

For each response, step ``j`` is the **mean token-level KL to uniform** within chunk
``j`` (non-overlapping chunks of ``chunk_size`` thinking-only tokens), using the same top‑k +
uniform-tail completion as ``abstain_kl.py`` / ``abstain_step_kl_experiment.py``.

Reads ``<outputs_dir>/<dataset>/<model>/result_*.json`` and writes one combined figure to
``abstaining_validation_plot/<dataset>/<model_safe>_val_step_kl.png``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from abstain_experiment_common import (  # noqa: E402
    load_results_flat_dir,
    sanitize_filename_component,
    stratified_val_test_split_fixed_size as stratified_val_test_split,
)
from abstain_kl import (  # noqa: E402
    filter_usable_examples_kl,
    token_kls_thinking_only,
)
from abstain_step_entropy import (  # noqa: E402
    chunk_step_means,
    val_step_mean_var_count_from_step_lists,
)

VALIDATION_PLOT_ROOT = "abstaining_validation_plot"


def plot_val_step_kl_combined(
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
            label="Mean chunk KL(U‖p) (correct)",
        )
    if np.any(mask_i):
        ax0.plot(
            j[mask_i],
            mean_incorrect[mask_i],
            color="tab:orange",
            linewidth=2,
            label="Mean chunk KL(U‖p) (incorrect)",
        )
    ax0.set_ylabel("Mean KL(U‖p) per chunk\n(within chunk, thinking tokens)")
    ax0.set_title(
        "(1) Mean chunk KL(U‖p) vs step — validation; per-step mean over responses that reach that step"
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
            label="Var(chunk mean KL) (correct)",
        )
    if np.any(mask_vi):
        ax1.plot(
            j[mask_vi],
            var_incorrect[mask_vi],
            color="tab:orange",
            linewidth=2,
            label="Var(chunk mean KL) (incorrect)",
        )
    ax1.set_ylabel("Variance of chunk mean KL(U‖p)")
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
        f"Validation — per-chunk mean KL(U‖p) (step KL) | chunk_size={chunk_size} tokens/chunk"
    )
    if title_suffix:
        supt = f"{title_suffix}\n{supt}"
    fig.suptitle(supt, fontsize=12, y=1.02)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot validation per-chunk mean KL(U‖p) vs step (correct vs incorrect)"
    )
    p.add_argument(
        "--outputs_dir",
        type=str,
        default="outputs",
        help="Root containing <dataset>/<model>/result_*.json (default: outputs)",
    )
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset folder name (e.g. gpqa)",
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model folder name under the dataset directory",
    )
    p.add_argument(
        "--vocab_size",
        type=int,
        required=True,
        help="Full vocabulary size V (same as step_kl abstention; see README / vocab_map)",
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
        "--title",
        type=str,
        default=None,
        help="Optional extra title line (default: use --model)",
    )
    args = p.parse_args()

    results_dir = os.path.join(args.outputs_dir, args.dataset, args.model)
    data = load_results_flat_dir(results_dir)
    print(f"Loaded {len(data)} files from {results_dir}")

    n_raw = len(data)
    data = filter_usable_examples_kl(data)
    if len(data) != n_raw:
        print(
            f"Dropped {n_raw - len(data)} examples without aligned tokens/top_logprobs; "
            f"{len(data)} remain."
        )
    if len(data) <= args.val_size:
        raise SystemExit(
            f"Need more than val_size={args.val_size} usable examples after filtering, "
            f"got {len(data)}"
        )

    val_data, _test_discard = stratified_val_test_split(data, args.val_size, args.seed)
    print(f"Using validation only: N_val={len(val_data)} (test split discarded for this plot)")

    vocab_size = int(args.vocab_size)
    correct_steps: list[list[float]] = []
    incorrect_steps: list[list[float]] = []
    for d in val_data:
        kl_seq = token_kls_thinking_only(d, vocab_size)
        steps = chunk_step_means(kl_seq, args.chunk_size)
        if d.get("is_correct"):
            correct_steps.append(steps)
        else:
            incorrect_steps.append(steps)

    mean_corr, mean_inc, var_corr, var_inc, n_corr, n_inc = (
        val_step_mean_var_count_from_step_lists(correct_steps, incorrect_steps)
    )

    out_dir = Path(VALIDATION_PLOT_ROOT) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = sanitize_filename_component(args.model)
    output_path = str(out_dir / f"{safe_model}_val_step_kl.png")
    title_suffix = args.title if args.title is not None else args.model

    plot_val_step_kl_combined(
        mean_corr,
        mean_inc,
        var_corr,
        var_inc,
        n_corr,
        n_inc,
        args.chunk_size,
        output_path,
        title_suffix=title_suffix,
    )
    print(f"Saved combined plot (3 panels, shared x-axis): {output_path}")


if __name__ == "__main__":
    main()
