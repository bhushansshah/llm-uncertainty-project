#!/usr/bin/env python3
"""
Run abstention experiments for each (dataset, method) pair over model subfolders under
``outputs_dir/<dataset>/<model>/`` (see README). Delegates to the per-method scripts in
``scripts/``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SCRIPTS = _ROOT / "scripts"

METHOD_TO_SCRIPT = {
    "step_entropy": "abstain_step_entropy_experiment.py",
    "neg_logprob": "abstain_step_neg_logprob_experiment.py",
    "agg_entropy": "abstain_step_agg_entropy_experiment.py",
}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run abstention batch jobs: outputs/<dataset>/<model>/ → "
        "abstaining_results/<dataset>/<method>/ and abstaining_plots/<dataset>/<method>/"
    )
    p.add_argument(
        "--outputs_dir",
        type=str,
        required=True,
        help="Root directory (e.g. outputs) containing <dataset>/<model>/result_*.json",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset folder names (immediate children of --outputs_dir)",
    )
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model folder names to include. Omit to run every model found under each dataset.",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHOD_TO_SCRIPT.keys()),
        default=["step_entropy", "neg_logprob", "agg_entropy"],
        help="Which abstention pipelines to run (default: all three)",
    )
    p.add_argument(
        "--abstaining_results_dir",
        type=str,
        default="abstaining_results",
        help="Root for per-dataset/method CSVs (default: abstaining_results)",
    )
    p.add_argument(
        "--abstaining_plots_dir",
        type=str,
        default="abstaining_plots",
        help="Root for per-dataset/method validation plots (default: abstaining_plots)",
    )
    p.add_argument("--val_size", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_support_per_class", type=int, default=3)
    args = p.parse_args()

    outputs_root = os.path.abspath(args.outputs_dir)

    for dataset in args.datasets:
        ds_path = os.path.join(outputs_root, dataset)
        if not os.path.isdir(ds_path):
            print(f"SKIP: missing dataset directory: {ds_path}")
            continue

        for method in args.methods:
            script_path = _SCRIPTS / METHOD_TO_SCRIPT[method]
            if not script_path.is_file():
                print(f"SKIP: script not found: {script_path}")
                continue

            cmd: list[str] = [
                sys.executable,
                str(script_path),
                "--outputs_dir",
                outputs_root,
                "--dataset",
                dataset,
                "--abstaining_results_dir",
                args.abstaining_results_dir,
                "--abstaining_plots_dir",
                args.abstaining_plots_dir,
                "--val_size",
                str(args.val_size),
                "--seed",
                str(args.seed),
                "--min_support_per_class",
                str(args.min_support_per_class),
            ]
            if args.models:
                cmd.extend(["--models", *args.models])

            print("\n" + "=" * 72)
            print(f"dataset={dataset}  method={method}")
            print(" ".join(cmd))
            print("=" * 72 + "\n")

            r = subprocess.run(cmd, cwd=str(_ROOT))
            if r.returncode != 0:
                print(
                    f"Warning: exit code {r.returncode} for dataset={dataset} method={method}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()
