"""
Shared utilities for step-based abstention experiments and validation plots.

- Loading ``result_*.json`` directories
- Stratified train/validation splits (fraction or fixed validation count)
- The common **data-driven grid** used by step entropy, −log p, and step KL (same hyperparameter
  schedule; only the per-token series differs)

Scripts in ``scripts/`` should import from here instead of duplicating these helpers.
"""

from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split

from abstain_step_entropy import (
    abstention_f1,
    build_active_and_tau_with_min_support,
    chunk_step_means,
    should_abstain,
    val_step_means_and_counts_from_step_lists,
)


def load_results_flat_dir(results_dir: str) -> list[dict]:
    """Load all ``result_<idx>.json`` files under ``results_dir``, sorted by index."""
    pattern = os.path.join(results_dir, "result_*.json")
    files = sorted(
        glob.glob(pattern),
        key=lambda f: int(os.path.basename(f).split("_")[1].split(".")[0]),
    )
    out: list[dict] = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def val_count_from_fraction(n: int, val_fraction: float) -> int:
    """
    Validation set size from a fraction of ``n`` (usable pool size).

    ``val_fraction`` must be strictly between 0 and 1. The count is ``round(n * val_fraction)``
    clamped so there is at least one validation and one test example when ``n >= 2``.
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(
            f"val_fraction must be in (0, 1) exclusive, got {val_fraction}"
        )
    if n < 2:
        raise ValueError(f"Need at least 2 examples to split, got {n}")
    val_size = int(round(n * val_fraction))
    return max(1, min(n - 1, val_size))


def stratified_val_test_split_fraction(
    data: list[dict],
    val_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Stratified split: validation size = ``val_count_from_fraction(len(data), val_fraction)``."""
    n = len(data)
    val_size = val_count_from_fraction(n, val_fraction)
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


def stratified_val_test_split_fixed_size(
    data: list[dict],
    val_size: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Stratified split with a fixed validation set size (used by validation-only plot scripts)."""
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


def sanitize_filename_component(name: str) -> str:
    """Safe fragment for plot/CSV filenames derived from model or dataset folder names."""
    safe = "".join(c if c not in r'\/:*?"<>|' else "_" for c in name)
    return safe.strip() or "model"


def evaluate_split_cached(
    step_means_per_example: list[list[float]],
    labels: list[bool],
    active: np.ndarray,
    tau: np.ndarray,
    offset: float,
    ground_threshold: int,
) -> tuple[float, float, float]:
    flags: list[bool] = []
    for steps in step_means_per_example:
        flags.append(should_abstain(steps, active, tau, offset, ground_threshold))
    prec, rec, f1, _, _, _ = abstention_f1(flags, labels)
    return prec, rec, f1


def answer_accuracy_on_split(
    step_means_per_example: list[list[float]],
    labels: list[bool],
    active: np.ndarray,
    tau: np.ndarray,
    offset: float,
    ground_threshold: int,
) -> float:
    """Fraction of examples that are answered (not abstained) and correct."""
    n = len(labels)
    if n == 0:
        return 0.0
    n_ok = 0
    for steps, lab in zip(step_means_per_example, labels):
        if not should_abstain(steps, active, tau, offset, ground_threshold) and lab:
            n_ok += 1
    return n_ok / n


def chunk_sizes_from_thinking_lengths(lens: list[int]) -> list[int]:
    """50, 100, … up to 70th percentile of lengths (rounded up to multiple of 50)."""
    if not lens:
        return [50]
    p70 = float(np.percentile(lens, 70))
    max_cs = max(50, int(np.ceil(p70 / 50.0)) * 50)
    return list(range(50, max_cs + 1, 50))


def positive_gaps_for_delta_stats(
    mean_corr: np.ndarray,
    mean_inc: np.ndarray,
    n_corr: np.ndarray,
    n_inc: np.ndarray,
    min_support_per_class: int,
) -> list[float]:
    gaps: list[float] = []
    n = len(mean_corr)
    for j in range(n):
        if n_corr[j] < min_support_per_class or n_inc[j] < min_support_per_class:
            continue
        mc, mi = mean_corr[j], mean_inc[j]
        if np.isnan(mc) or np.isnan(mi) or mi <= mc:
            continue
        gaps.append(float(mi - mc))
    return gaps


def deltas_from_mu_sigma(mu: float, sigma: float, n: int = 20) -> list[float]:
    lo = mu - 2.5 * sigma
    hi = mu + 2.5 * sigma
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = 0.0, 0.02
    if hi <= lo:
        hi = lo + 1e-9
    return [float(x) for x in np.linspace(lo, hi, n)]


def offset_upper_bound_for_delta(
    delta: float,
    mean_corr: np.ndarray,
    mean_inc: np.ndarray,
    n_corr: np.ndarray,
    n_inc: np.ndarray,
    correct_steps: list[list[float]],
    min_support_per_class: int,
) -> float:
    """P95 of 2.5 × std(correct per-step value at j) over steps active for this δ."""
    vals: list[float] = []
    n = len(mean_corr)
    for j in range(n):
        if n_corr[j] < min_support_per_class or n_inc[j] < min_support_per_class:
            continue
        mc, mi = mean_corr[j], mean_inc[j]
        if np.isnan(mc) or np.isnan(mi):
            continue
        if mi <= mc:
            continue
        if (mi - mc) < delta:
            continue
        cvals = [s[j] for s in correct_steps if j < len(s)]
        if len(cvals) < 2:
            std = 0.0
        else:
            std = float(np.std(cvals, ddof=0))
        vals.append(2.5 * std)
    if not vals:
        return 0.0
    return float(np.percentile(vals, 95))


def run_step_signal_grid(
    val_data: list[dict],
    thinking_per_token: list[list[float]],
    save_csv: str | None,
    min_support_per_class: int,
) -> tuple[dict, list, list[tuple[int, float, np.ndarray]]]:
    """
    Validation grid for step-based abstention when the per-token signal is already computed.

    ``thinking_per_token[i]`` must align with ``val_data[i]`` (thinking-only token series before
    chunking). Chunk means, δ grid, offset grid, and τ records match
    ``abstain_step_entropy_experiment`` / neg_logprob / step_kl.

    Each grid row includes ``validation_accuracy``: (# non-abstained & correct on the validation
    split) / N_val, using the same abstention policy as for precision/recall/F1 on validation.
    """
    lens = [len(t) for t in thinking_per_token]
    chunk_sizes = chunk_sizes_from_thinking_lengths(lens)
    grounds = list(range(1, 21))

    labels = [bool(d.get("is_correct")) for d in val_data]

    correct_steps_by_cs: dict[int, list[list[float]]] = {}
    incorrect_steps_by_cs: dict[int, list[list[float]]] = {}
    for cs in chunk_sizes:
        csl, isl = [], []
        for d, seq in zip(val_data, thinking_per_token):
            steps = chunk_step_means(seq, cs)
            if d.get("is_correct"):
                csl.append(steps)
            else:
                isl.append(steps)
        correct_steps_by_cs[cs] = csl
        incorrect_steps_by_cs[cs] = isl

    step_means_flat_by_cs: dict[int, list[list[float]]] = {}
    for cs in chunk_sizes:
        step_means_flat_by_cs[cs] = [
            chunk_step_means(seq, cs) for seq in thinking_per_token
        ]

    best: dict | None = None
    rows: list[dict] = []
    tau_records: list[tuple[int, float, np.ndarray]] = []

    for chunk_size in chunk_sizes:
        mean_corr, mean_inc, n_corr, n_inc = val_step_means_and_counts_from_step_lists(
            correct_steps_by_cs[chunk_size],
            incorrect_steps_by_cs[chunk_size],
        )
        gaps = positive_gaps_for_delta_stats(
            mean_corr, mean_inc, n_corr, n_inc, min_support_per_class
        )
        if not gaps:
            mu, sigma = 0.0, 0.01
        else:
            mu = float(np.mean(gaps))
            sigma = float(np.std(gaps, ddof=0))
            if sigma <= 0 or not np.isfinite(sigma):
                sigma = max(abs(mu) * 0.01, 1e-6)
        deltas = deltas_from_mu_sigma(mu, sigma, 20)

        for delta in deltas:
            active, tau = build_active_and_tau_with_min_support(
                mean_corr, mean_inc, delta, n_corr, n_inc, min_support_per_class
            )
            tau_records.append((chunk_size, float(delta), np.array(tau, dtype=float)))

            p95 = offset_upper_bound_for_delta(
                delta,
                mean_corr,
                mean_inc,
                n_corr,
                n_inc,
                correct_steps_by_cs[chunk_size],
                min_support_per_class,
            )
            if p95 <= 0 or not np.isfinite(p95):
                offsets = [0.0] * 20
            else:
                offsets = [float(x) for x in np.linspace(0.0, p95, 20)]

            for offset in offsets:
                for g in grounds:
                    prec, rec, f1 = evaluate_split_cached(
                        step_means_flat_by_cs[chunk_size],
                        labels,
                        active,
                        tau,
                        offset,
                        g,
                    )
                    validation_accuracy = answer_accuracy_on_split(
                        step_means_flat_by_cs[chunk_size],
                        labels,
                        active,
                        tau,
                        offset,
                        g,
                    )
                    row = {
                        "chunk_size": chunk_size,
                        "delta": delta,
                        "offset": offset,
                        "ground_threshold": g,
                        "min_support_per_class": min_support_per_class,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "validation_accuracy": validation_accuracy,
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

    return (best or {}), rows, tau_records
