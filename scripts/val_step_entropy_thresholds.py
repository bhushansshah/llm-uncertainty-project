#!/usr/bin/env python3
"""
Print validation-derived per-step τ + offset and active flags for step-entropy abstention.

Uses the same data path as scripts/abstain_step_entropy_experiment.py: load result_*.json,
filter usable entropy examples, stratified val split, then fit τ and active on validation only.
``ground_threshold`` does not change τ or active; it only enters ``should_abstain``. Optional
``--histogram`` prints a distribution of **trigger step indices** (first step where the running
count of active exceedances reaches ``ground_threshold``) among abstained validation examples.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from abstain_experiment_common import (  # noqa: E402
    load_results_flat_dir,
    stratified_val_test_split_fraction as stratified_val_test_split,
)
from abstain_step_entropy import (  # noqa: E402
    build_active_and_tau_with_min_support,
    chunk_step_means,
    filter_usable_examples_entropy,
    first_step_index_abstention_prefix,
    should_abstain,
    token_entropies_thinking_only,
    val_step_means_and_counts_from_step_lists,
)


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


def _fmt_threshold(x: float) -> str:
    if np.isnan(x):
        return "N/A"
    return f"{float(x):.8g}"


def abstained_validation_abstention_trigger_steps(
    val_data: list[dict],
    *,
    chunk_size: int,
    delta: float,
    offset: float,
    ground_threshold: int,
    min_support_per_class: int,
) -> list[int]:
    """
    Fit τ and active on ``val_data``. For each example that abstains (``should_abstain`` on the
    full chunk-mean sequence), record the **chunk step index** at which abstention is first
    satisfied in reading order: the smallest ``j`` such that at least ``ground_threshold``
    active steps among ``0..j`` have mean above ``τ+offset`` (see
    ``first_step_index_abstention_prefix`` in ``abstain_step_entropy``).

    This is not total response length; it is *where* in the step sequence the threshold count
    is first reached.
    """
    mean_corr, mean_inc, n_corr, n_inc = val_step_entropy_stats(val_data, chunk_size)
    active, tau = build_active_and_tau_with_min_support(
        mean_corr,
        mean_inc,
        float(delta),
        n_corr,
        n_inc,
        int(min_support_per_class),
    )
    off = float(offset)
    g = int(ground_threshold)
    out: list[int] = []
    for d in val_data:
        steps = chunk_step_means(token_entropies_thinking_only(d), chunk_size)
        if not should_abstain(steps, active, tau, off, g):
            continue
        j = first_step_index_abstention_prefix(steps, active, tau, off, g)
        if j is not None:
            out.append(j)
    return out


def print_abstained_validation_step_count_histogram(
    val_data: list[dict],
    *,
    chunk_size: int,
    delta: float,
    offset: float,
    ground_threshold: int,
    min_support_per_class: int,
) -> list[int]:
    """
    For each validation example that abstains, histogram the **step index** where the running
    count of active exceedances first reaches ``ground_threshold`` (prefix / reading order).
    """
    counts = abstained_validation_abstention_trigger_steps(
        val_data,
        chunk_size=chunk_size,
        delta=delta,
        offset=offset,
        ground_threshold=ground_threshold,
        min_support_per_class=min_support_per_class,
    )
    print()
    print(
        "=== Histogram: step index where abstention first triggers (abstained val examples) ==="
    )
    print(
        f"(first step j where ≥{ground_threshold} active steps in 0..j exceed τ+offset; "
        f"min_support_per_class={min_support_per_class})"
    )
    n_val = len(val_data)
    n_abs = len(counts)
    print(f"validation examples: {n_val}, abstained: {n_abs}")
    if not counts:
        print("No abstained examples; empty histogram.")
        return counts

    arr = np.asarray(counts, dtype=int)
    print(
        f"trigger step index — min: {arr.min()}, "
        f"median: {float(np.median(arr)):.1f}, max: {arr.max()}, mean: {arr.mean():.3f}"
    )
    uniq, freq = np.unique(arr, return_counts=True)
    mx = int(freq.max())
    print("step_idx  count  bar")
    for u, c in zip(uniq, freq):
        bar = "#" * max(1, int(40 * c / mx)) if mx else ""
        print(f"{int(u):8d}  {int(c):5d}  {bar}")
    return counts


def main() -> None:
    p = argparse.ArgumentParser(
        description="Print τ+offset per step and active[] from validation (step entropy)"
    )
    p.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing result_0.json, ...",
    )
    p.add_argument(
        "--val_fraction",
        type=float,
        default=0.3,
        help="Fraction of usable examples in validation (default: 0.3)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for stratified split")
    p.add_argument("--chunk_size", type=int, required=True)
    p.add_argument("--delta", type=float, required=True)
    p.add_argument("--offset", type=float, required=True)
    p.add_argument(
        "--ground_threshold",
        type=int,
        required=True,
        help="Min active steps above τ+offset to abstain (does not affect τ/active; used with --histogram)",
    )
    p.add_argument(
        "--histogram",
        action="store_true",
        default=True,
        help="Histogram of first trigger step index (prefix) for validation examples that abstain",
    )
    p.add_argument(
        "--min_support_per_class",
        "--main-support",
        type=int,
        default=3,
        dest="min_support_per_class",
        help="Min correct and min incorrect val responses at a step for activation (main support)",
    )
    args = p.parse_args()

    if not (0.0 < args.val_fraction < 1.0):
        p.error("--val_fraction must be strictly between 0 and 1")

    if args.chunk_size < 1:
        p.error("--chunk_size must be >= 1")

    if args.min_support_per_class < 1:
        p.error("--min_support_per_class must be >= 1")

    data = load_results_flat_dir(args.results_dir)
    n_raw = len(data)
    data = filter_usable_examples_entropy(data)
    n_usable = len(data)
    if n_usable < 2:
        raise SystemExit(
            f"Need at least 2 usable examples after filtering, got {n_usable} "
            f"(loaded {n_raw} files from {args.results_dir})"
        )

    val_data, _test_data = stratified_val_test_split(data, args.val_fraction, args.seed)

    mean_corr, mean_inc, n_corr, n_inc = val_step_entropy_stats(
        val_data, args.chunk_size
    )
    active, tau = build_active_and_tau_with_min_support(
        mean_corr,
        mean_inc,
        float(args.delta),
        n_corr,
        n_inc,
        int(args.min_support_per_class),
    )

    offset = float(args.offset)
    effective = tau + offset

    n_steps = len(tau)
    n_active = int(np.sum(active))

    print("=== Step-entropy thresholds (validation fit) ===")
    print(f"results_dir:          {args.results_dir}")
    print(f"n_usable (filtered):  {n_usable}")
    print(f"n_validation:         {len(val_data)}")
    print(f"val_fraction:         {args.val_fraction}")
    print(f"seed:                 {args.seed}")
    print(f"chunk_size:           {args.chunk_size}")
    print(f"delta:                {args.delta}")
    print(f"offset:               {offset}")
    print(f"ground_threshold:     {args.ground_threshold}  (abstain rule only; does not affect τ/active)")
    print(f"min_support_per_class: {args.min_support_per_class}")
    print(f"n_steps:              {n_steps}")
    print(f"n_active_steps:       {n_active}")
    if n_active == 0:
        print(
            "\nWarning: no step met activation criteria (active is all False; τ is all NaN)."
        )
    print()
    print("Per-step effective cutoff τ[j] + offset (N/A where step is inactive / τ undefined):")
    print("  step_index  tau_plus_offset  active")
    for j in range(n_steps):
        print(f"  {j:10d}  {_fmt_threshold(effective[j]):>15}  {bool(active[j])}")
    print()
    print("tau_plus_offset (same order, one value per line):")
    for j in range(n_steps):
        print(_fmt_threshold(effective[j]))
    print()
    print("active (0/1, same step order):")
    print("".join("1" if active[j] else "0" for j in range(n_steps)))

    if args.histogram:
        print_abstained_validation_step_count_histogram(
            val_data,
            chunk_size=args.chunk_size,
            delta=float(args.delta),
            offset=offset,
            ground_threshold=int(args.ground_threshold),
            min_support_per_class=int(args.min_support_per_class),
        )


if __name__ == "__main__":
    main()
