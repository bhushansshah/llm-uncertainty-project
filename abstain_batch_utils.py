"""Shared helpers for abstention batch runs: outputs/<dataset>/<model>/result_*.json layout."""

from __future__ import annotations

import glob
import os
from pathlib import Path

# Subdirectory names under abstaining_results/<dataset>/ and abstaining_plots/<dataset>/
METHOD_STEP_ENTROPY = "step_entropy"
METHOD_NEG_LOGPROB = "neg_logprob"
METHOD_AGG_ENTROPY = "agg_entropy"


def discover_models_in_dataset(
    dataset_dir: str | Path,
    model_names: list[str] | None,
) -> list[tuple[str, str]]:
    """
    List (model_folder_name, absolute_path) for model dirs under ``dataset_dir`` that
    contain at least one ``result_*.json``.

    If ``model_names`` is non-empty, only those names are considered (in listed order).
    Missing dirs or dirs without results are skipped with a printed message.
    If ``model_names`` is empty, every immediate subdirectory with ``result_*.json`` is
    included (sorted by name).
    """
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {dataset_dir}")

    def has_results(p: Path) -> bool:
        return bool(glob.glob(str(p / "result_*.json")))

    out: list[tuple[str, str]] = []
    if model_names:
        for name in model_names:
            p = root / name
            if not p.is_dir():
                print(f"SKIP: no directory {p}")
                continue
            if not has_results(p):
                print(f"SKIP: no result_*.json in {p}")
                continue
            out.append((name, str(p.resolve())))
        return out

    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if has_results(p):
            out.append((p.name, str(p.resolve())))
    return out
