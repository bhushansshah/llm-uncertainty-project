#!/usr/bin/env python3
"""
Summarize a single best-of-N experiment folder: load ``result_*.json``, aggregate metrics over
all questions, histogram of total tokens per question, write ``evaluation.json`` and
``total_tokens_histogram.png`` to ``--output-dir``.

Non-abstained count per question: entries in ``responses`` with ``not r.get("is_abstained")``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _result_index(path: str) -> int:
    base = os.path.basename(path)
    m = re.match(r"result_(\d+)\.json$", base)
    if not m:
        return -1
    return int(m.group(1))


def load_series(results_dir: str) -> dict[int, dict]:
    """Map result_index -> JSON object."""
    pattern = os.path.join(os.path.abspath(results_dir), "result_*.json")
    files = sorted(glob.glob(pattern), key=_result_index)
    out: dict[int, dict] = {}
    for fp in files:
        ix = _result_index(fp)
        if ix < 0:
            continue
        with open(fp, encoding="utf-8") as f:
            out[ix] = json.load(f)
    return out


def _count_non_abstained_responses(responses: list | None) -> int:
    """Responses with ``is_abstained`` false or missing (treated as not abstained)."""
    if not responses:
        return 0
    return sum(1 for r in responses if r.get("is_abstained"))


def summarize_results(indices: list[int], data: list[dict], *, label: str = "") -> dict:
    n = len(data)
    correct = [bool(d.get("is_selected_correct")) for d in data]
    tokens = [int(d.get("total_tokens", 0) or 0) for d in data]
    times = [float(d.get("total_time_seconds", 0.0) or 0.0) for d in data]
    n_non_ab = [_count_non_abstained_responses(d.get("responses")) for d in data]

    acc = float(np.mean(correct)) if n else 0.0
    out: dict = {
        "n_questions": n,
        "accuracy": acc,
        "n_correct": int(sum(correct)),
        "mean_total_tokens_per_question": float(np.mean(tokens)) if n else 0.0,
        "std_total_tokens_per_question": float(np.std(tokens, ddof=0)) if n else 0.0,
        "mean_total_time_seconds_per_question": float(np.mean(times)) if n else 0.0,
        "std_total_time_seconds_per_question": float(np.std(times, ddof=0)) if n else 0.0,
        "mean_num_non_abstained_responses_per_question": float(np.mean(n_non_ab))
        if n
        else 0.0,
        "std_num_non_abstained_responses_per_question": float(np.std(n_non_ab, ddof=0))
        if n
        else 0.0,
        "per_question": [
            {
                "result_index": int(indices[i]),
                "is_selected_correct": correct[i],
                "total_tokens": tokens[i],
                "total_time_seconds": times[i],
                "num_non_abstained_responses": n_non_ab[i],
            }
            for i in range(n)
        ],
    }
    if label:
        out["label"] = label
    return out


def histogram_payload_single(tokens: list[float], bins: int) -> dict:
    all_t = np.asarray(tokens, dtype=float)
    if all_t.size == 0:
        return {"bin_edges": [], "counts": []}
    lo, hi = float(all_t.min()), float(all_t.max())
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    edges = np.linspace(lo, hi, int(bins) + 1)
    counts, _ = np.histogram(all_t, bins=edges)
    return {
        "bin_edges": edges.tolist(),
        "counts": counts.astype(int).tolist(),
    }


def plot_histogram_single(
    tokens: list[float],
    bins: int,
    out_path: Path,
    *,
    title: str = "Total tokens per question",
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("Install matplotlib: pip install matplotlib") from e

    all_t = np.asarray(tokens, dtype=float)
    if all_t.size == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    lo, hi = float(all_t.min()), float(all_t.max())
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    edges = np.linspace(lo, hi, int(bins) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(all_t, bins=edges, edgecolor="black", linewidth=0.5, alpha=0.75)
    ax.set_xlabel("Total tokens per question")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Summarize one BON result folder (result_*.json) → evaluation.json + histogram"
    )
    p.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing result_*.json from a BON run",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Where to write evaluation.json and total_tokens_histogram.png",
    )
    p.add_argument(
        "--histogram-bins",
        type=int,
        default=20,
        help="Number of bins for total-token histogram",
    )
    p.add_argument(
        "--label",
        type=str,
        default="",
        help="Optional label stored in summary JSON under key \"label\"",
    )
    p.add_argument(
        "--plot-title",
        type=str,
        default="Total tokens per question",
        help="Title for the histogram figure",
    )
    args = p.parse_args()

    if args.histogram_bins < 1:
        p.error("--histogram-bins must be >= 1")

    by_idx = load_series(args.results_dir)
    if not by_idx:
        raise SystemExit(f"No result_*.json found under {args.results_dir}")

    indices = sorted(by_idx.keys())
    data = [by_idx[i] for i in indices]

    summary = summarize_results(indices, data, label=args.label.strip())

    token_list = [float(summary["per_question"][i]["total_tokens"]) for i in range(len(indices))]
    hist = histogram_payload_single(token_list, args.histogram_bins)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluation = {
        "summary": summary,
        "result_indices": indices,
        "histogram_bins": args.histogram_bins,
        "histogram": hist,
        "paths": {
            "results_dir": os.path.abspath(args.results_dir),
            "output_dir": str(out_dir.resolve()),
        },
    }

    eval_path = out_dir / "evaluation.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)

    plot_path = out_dir / "total_tokens_histogram.png"
    plot_histogram_single(
        token_list,
        args.histogram_bins,
        plot_path,
        title=args.plot_title,
    )

    print(f"Wrote {eval_path}", flush=True)
    print(f"Wrote {plot_path}", flush=True)


if __name__ == "__main__":
    main()
