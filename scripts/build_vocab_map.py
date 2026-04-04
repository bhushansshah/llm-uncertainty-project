#!/usr/bin/env python3
"""Build vocab_map.json: model folder name -> tokenizer vocabulary size (len(tokenizer)).

Usage: python scripts/build_vocab_map.py --gpqa-dir outputs/gpqa [--output vocab_map.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _list_model_subdirs(gpqa_dir: Path) -> list[str]:
    if not gpqa_dir.is_dir():
        raise FileNotFoundError(f"gpqa directory does not exist or is not a directory: {gpqa_dir}")
    names: list[str] = []
    for entry in sorted(gpqa_dir.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            names.append(entry.name)
    return names


def _folder_to_hf_id_heuristic(folder_name: str) -> str:
    if "_" not in folder_name:
        return folder_name
    return folder_name.replace("_", "/", 1)


def _resolve_hf_id(
    folder_name: str,
    model_dir: Path,
    overrides: dict[str, str],
) -> str:
    if folder_name in overrides:
        return overrides[folder_name]
    config_path = model_dir / "config.json"
    if config_path.is_file():
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            args = cfg.get("arguments")
            if isinstance(args, dict) and args.get("model"):
                return str(args["model"])
        except (json.JSONDecodeError, OSError):
            pass
    return _folder_to_hf_id_heuristic(folder_name)


def _warn_config_vs_tokenizer(hf_id: str, tokenizer_len: int) -> None:
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
        v = getattr(config, "vocab_size", None)
        if v is not None and int(v) != tokenizer_len:
            print(
                f"  warning: AutoConfig.vocab_size={v} != len(tokenizer)={tokenizer_len} for {hf_id!r}",
                file=sys.stderr,
            )
    except Exception as e:
        print(f"  warning: could not load AutoConfig for comparison: {e}", file=sys.stderr)


def main() -> int:
    default_out = _REPO_ROOT / "vocab_map.json"
    p = argparse.ArgumentParser(
        description="Map GPQA model subfolder names to len(tokenizer) and write JSON."
    )
    p.add_argument(
        "--gpqa-dir",
        type=Path,
        default=Path("outputs/gpqa"),
        help="Directory whose immediate subdirs are model folder names (default: outputs/gpqa)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=default_out,
        help=f"Output JSON path (default: {default_out})",
    )
    p.add_argument(
        "--override",
        type=Path,
        default=None,
        help='Optional JSON {"<folder_name>": "org/model-id", ...} for manual HF ids',
    )
    p.set_defaults(strict=True)
    p.add_argument(
        "--no-strict",
        action="store_false",
        dest="strict",
        help="Write partial vocab_map and exit 0 even when some models fail",
    )
    p.add_argument(
        "--skip-config-check",
        action="store_true",
        help="Do not compare AutoConfig.vocab_size to len(tokenizer)",
    )
    args = p.parse_args()

    gpqa_dir = args.gpqa_dir.resolve()
    overrides: dict[str, str] = {}
    if args.override is not None:
        with open(args.override, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise SystemExit("--override JSON must be an object")
        overrides = {str(k): str(v) for k, v in raw.items()}

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            "transformers is required. Install with: pip install transformers"
        ) from e

    subdirs = _list_model_subdirs(gpqa_dir)
    if not subdirs:
        print(f"No model subdirectories under {gpqa_dir}", file=sys.stderr)
        return 1

    result: dict[str, int] = {}
    failures: list[tuple[str, str]] = []

    for name in subdirs:
        model_dir = gpqa_dir / name
        hf_id = _resolve_hf_id(name, model_dir, overrides)
        print(f"{name} -> {hf_id!r}")
        try:
            tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        except Exception as e:
            failures.append((name, f"AutoTokenizer.from_pretrained: {e}"))
            continue
        n = len(tok)
        result[name] = n
        if not args.skip_config_check:
            _warn_config_vs_tokenizer(hf_id, n)

    out_path = args.output
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {k: result[k] for k in sorted(result.keys())}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2)
        f.write("\n")
    print(f"Wrote {len(ordered)} entries to {out_path}")

    if failures:
        print("\nFailures:", file=sys.stderr)
        for name, msg in failures:
            print(f"  {name}: {msg}", file=sys.stderr)
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
