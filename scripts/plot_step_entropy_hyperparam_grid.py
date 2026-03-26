#!/usr/bin/env python3
"""
Visualize validation F1 over the step-entropy abstention grid from
``abstain_step_entropy_experiment.py --output_csv`` or the combined GPQA
``abstaining_results/gpqa/grid.csv``.

The grid has four tunable hyperparameters (chunk_size, delta, noise, ground_threshold) plus
``min_support_per_class`` (fixed for a given CSV run). This script produces **two** figures
per run (or **per model** when the CSV has a ``model_name`` column):

1. **Summary heatmap** — For each (chunk_size, delta), shows the **best F1** achievable over
   all (noise, ground_threshold) in that slice.

2. **Faceted scatter** — One subplot per **delta**. Each point is **one full combination**;
   **x** = chunk_size, **y** = validation F1, **color** = noise, **marker size** = ground_threshold.

**GPQA combined CSV** (multiple ``model_name`` values): writes
``<output_dir>/<SafeModelName>_summary.png`` and ``_detail.png`` for each model (default
``output_dir`` = ``abstaining_plots/gpqa``).

**Single-model CSV** (no ``model_name`` column): writes ``<prefix>_summary.png`` and
``<prefix>_detail.png`` as before.

Usage:
  python scripts/plot_step_entropy_hyperparam_grid.py --csv grid_results.csv \\
    --output_prefix abstention_grid_viz

  python scripts/plot_step_entropy_hyperparam_grid.py \\
    --csv abstaining_results/gpqa/grid.csv --output_dir abstaining_plots/gpqa
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REQUIRED_COLS = ("chunk_size", "delta", "noise", "ground_threshold", "f1")


def _sanitize_filename_component(name: str) -> str:
    safe = "".join(c if c not in r'\/:*?"<>|' else "_" for c in name)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe.strip() or "model"


def load_grid_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns {missing}. Found: {list(df.columns)}")
    for c in ("chunk_size", "ground_threshold"):
        df[c] = df[c].astype(int)
    for c in ("delta", "noise", "f1"):
        df[c] = df[c].astype(float)
    if "model_name" in df.columns:
        df["model_name"] = df["model_name"].astype(str)
    return df


def plot_summary_heatmap(
    df: pd.DataFrame,
    out_path: str,
    title_suffix: str | None = None,
) -> None:
    """Max F1 over (noise, ground_threshold) for each (chunk_size, delta)."""
    agg = (
        df.groupby(["chunk_size", "delta"], as_index=False)["f1"]
        .max()
        .rename(columns={"f1": "f1_max"})
    )
    pivot = agg.pivot(index="delta", columns="chunk_size", values="f1_max")
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    data = pivot.values
    im = ax.imshow(
        data,
        aspect="auto",
        cmap="viridis",
        vmin=float(np.nanmin(data)),
        vmax=float(np.nanmax(data)),
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Max validation F1 (over noise, ground_threshold)")

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([f"{float(r):.2f}" for r in pivot.index])

    ax.set_xlabel("chunk_size")
    ax.set_ylabel("delta (min gap incorrect vs correct mean entropy)")
    base = "Step-entropy grid: best validation F1 per (chunk_size, δ)"
    ax.set_title(f"{title_suffix}\n{base}" if title_suffix else base)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_faceted_scatter(
    df: pd.DataFrame,
    out_path: str,
    title_suffix: str | None = None,
) -> None:
    """One row per delta; points = full combos; color=noise, size=ground_threshold."""
    deltas = sorted(df["delta"].unique())
    n = len(deltas)
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(12, 2.2 * n),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if n == 1:
        axes = np.array([axes])

    noise_min = float(df["noise"].min())
    noise_max = float(df["noise"].max())
    norm = mcolors.Normalize(vmin=noise_min, vmax=noise_max)
    cmap = plt.cm.viridis

    g_min = int(df["ground_threshold"].min())
    g_max = int(df["ground_threshold"].max())

    def size_for_g(g: int) -> float:
        g = float(g)
        return 15.0 + (g - g_min) / max(g_max - g_min, 1) * 120.0

    for ax, dval in zip(axes, deltas):
        sub = df[df["delta"] == dval]
        x = sub["chunk_size"].values
        y = sub["f1"].values
        noise = sub["noise"].values
        colors = [cmap(norm(v)) for v in noise]
        sizes = [size_for_g(int(g)) for g in sub["ground_threshold"].values]
        ax.scatter(
            x,
            y,
            c=colors,
            s=sizes,
            alpha=0.45,
            edgecolors="none",
        )
        ax.set_ylabel(f"F1\n(δ={dval:.2f})", fontsize=9)
        ax.grid(True, alpha=0.25)
        if len(x):
            ax.set_xlim(float(x.min()) - 25, float(x.max()) + 25)

    axes[-1].set_xlabel("chunk_size")
    axes[0].set_ylim(max(0, float(df["f1"].min()) - 0.05), min(1.05, float(df["f1"].max()) + 0.05))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("noise (added to τ)")

    uniq_g = sorted(df["ground_threshold"].unique())
    if len(uniq_g) <= 3:
        show_gs = [int(x) for x in uniq_g]
    else:
        show_gs = [int(uniq_g[0]), int(uniq_g[len(uniq_g) // 2]), int(uniq_g[-1])]
    size_legend = [
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markersize=np.sqrt(size_for_g(g)) / 2,
            markerfacecolor="0.45",
            markeredgecolor="none",
            label=f"ground_threshold = {g}",
        )
        for g in show_gs
    ]
    axes[0].legend(
        handles=size_legend,
        loc="lower right",
        fontsize=8,
        title="Marker size",
        framealpha=0.9,
    )

    supt = (
        "Every point = one (chunk_size, δ, noise, ground_threshold) combo. "
        "Color = noise, size = ground_threshold."
    )
    if title_suffix:
        supt = f"{title_suffix}\n{supt}"
    fig.suptitle(supt, fontsize=11, y=1.01)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot validation F1 vs step-entropy hyperparameter grid (from --output_csv or GPQA grid.csv)"
    )
    p.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Grid CSV (single-model or GPQA combined grid with model_name)",
    )
    p.add_argument(
        "--output_prefix",
        type=str,
        default="step_entropy_hyperparam_grid",
        help="Single-model mode: write <prefix>_summary.png and <prefix>_detail.png (ignored when model_name has multiple values)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="abstaining_plots/gpqa",
        help="Multi-model mode: directory for per-model PNGs (default: abstaining_plots/gpqa)",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional extra title line (single-model mode only; multi-model uses each model name)",
    )
    args = p.parse_args()

    df = load_grid_csv(args.csv)
    if len(df) == 0:
        raise SystemExit("CSV has no rows.")

    if "model_name" in df.columns:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        models = sorted(df["model_name"].unique(), key=str)
        for model in models:
            sub = df[df["model_name"] == model].drop(columns=["model_name"])
            if len(sub) == 0:
                continue
            safe = _sanitize_filename_component(str(model))
            summary_path = out_dir / f"{safe}_summary.png"
            detail_path = out_dir / f"{safe}_detail.png"
            plot_summary_heatmap(sub, str(summary_path), title_suffix=str(model))
            plot_faceted_scatter(sub, str(detail_path), title_suffix=str(model))
            print(f"Wrote {model}: {summary_path}")
            print(f"           {detail_path}")
        return

    # Single-model CSV (no model_name column)

    prefix = Path(args.output_prefix)
    parent = prefix.parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
    stem = prefix.name

    summary_path = prefix.parent / f"{stem}_summary.png"
    detail_path = prefix.parent / f"{stem}_detail.png"

    plot_summary_heatmap(df, str(summary_path), title_suffix=args.title)
    plot_faceted_scatter(df, str(detail_path), title_suffix=args.title)

    print(f"Wrote summary heatmap: {summary_path}")
    print(f"Wrote faceted scatter: {detail_path}")


if __name__ == "__main__":
    main()
