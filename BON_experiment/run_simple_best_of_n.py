#!/usr/bin/env python3
"""
Simple best-of-N (no abstention): for each test question, generate ``N`` full streaming
responses with logprobs, compute mean thinking-token entropy as **uncertainty**, pick the
response with the **lowest** uncertainty, and record correctness vs gold.

Uses the same data contract as ``run_best_of_n.py``: load ``result_*.json`` from
``--calibration-results-dir``, ``filter_usable_examples_entropy``, stratified val/test split;
**test** rows drive new generations (stored ``response`` ignored).

Outputs: ``simple_best_of_N/<dataset_name>/<model>/config.json`` and ``result_{idx}.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

from abstain_experiment_common import (  # noqa: E402
    load_results_flat_dir,
    sanitize_filename_component,
    stratified_val_test_split_fraction,
)
from abstain_step_entropy import filter_usable_examples_entropy  # noqa: E402
from generate_responses_verbalized import extract_answer, is_correct_answer  # noqa: E402

from BON_experiment.run_best_of_n import (  # noqa: E402
    DEFAULT_PROMPT,
    _json_default,
    _uncertainty_sort_key,
    build_client,
    letter_from_parsed,
    split_question_and_options,
)
from BON_experiment.streaming_abstain import stream_chat_collect_only  # noqa: E402


def run_one_simple_attempt(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    top_logprobs: int,
) -> dict:
    extra: dict = {}
    if top_k > 0:
        extra["top_k"] = top_k

    create_kw: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }
    if extra:
        create_kw["extra_body"] = extra
    stream = client.chat.completions.create(**create_kw)

    raw = stream_chat_collect_only(stream)
    full_text = "".join(raw["tokens"])
    raw["parsed_option"] = letter_from_parsed(extract_answer(full_text))
    return raw


def main() -> None:
    p = argparse.ArgumentParser(
        description="Simple best-of-N: full N generations, select lowest thinking entropy"
    )
    p.add_argument(
        "--calibration-results-dir",
        type=str,
        required=True,
        help="Directory with result_*.json (same pool as abstaining BON; test split is used)",
    )
    p.add_argument("--val-fraction", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=4, help="Number of responses per question")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-32B")
    p.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("BASE_URL", "http://localhost:8000/v1"),
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("API_KEY", "EMPTY"),
    )
    p.add_argument("--max-tokens", type=int, default=26000)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="If >0, passed as extra_body top_k (vLLM). Use 0 for Hugging Face router.",
    )
    p.add_argument("--top-logprobs", type=int, default=10)
    p.add_argument("--prompt", type=str, default="", help="Template with {question} and {option}")
    p.add_argument("--dataset-name", type=str, default="gpqa")
    p.add_argument(
        "--output-root",
        type=str,
        default=str(_ROOT / "simple_best_of_N_outputs"),
        help="Root directory (default: simple_best_of_N_outputs under repo root)",
    )
    p.add_argument("--test-limit", type=int, default=0, help="If >0, only first K test questions")
    p.add_argument("--system", type=str, default="", help="Optional system message")
    args = p.parse_args()

    if not (0.0 < args.val_fraction < 1.0):
        p.error("--val-fraction must be between 0 and 1")
    if args.n < 1:
        p.error("--n must be >= 1")

    prompt_template = args.prompt.strip() or DEFAULT_PROMPT

    raw_data = load_results_flat_dir(args.calibration_results_dir)
    usable = filter_usable_examples_entropy(raw_data)
    if len(usable) < 2:
        raise SystemExit(f"Need >= 2 usable examples after filtering, got {len(usable)}")

    val_data, test_data = stratified_val_test_split_fraction(
        usable, args.val_fraction, args.seed
    )

    top_lp = max(1, min(int(args.top_logprobs), 20))

    client = build_client(args.base_url, args.api_key)
    model_safe = sanitize_filename_component(args.model)
    ds_safe = sanitize_filename_component(args.dataset_name)
    out_dir = Path(args.output_root) / ds_safe / model_safe
    out_dir.mkdir(parents=True, exist_ok=True)

    config_block = {
        "mode": "simple_best_of_n",
        "calibration_results_dir": os.path.abspath(args.calibration_results_dir),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "n": args.n,
        "model": args.model,
        "base_url": args.base_url,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "top_logprobs": top_lp,
        "dataset_name": args.dataset_name,
        "n_val": len(val_data),
        "n_test": len(test_data),
        "n_usable_calibration": len(usable),
    }

    config_path = out_dir / "config.json"
    config_path.write_text(
        json.dumps(config_block, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"Wrote {config_path}", flush=True)

    limit = args.test_limit if args.test_limit > 0 else len(test_data)

    for test_idx in range(min(limit, len(test_data))):
        row = test_data[test_idx]
        q_full = str(row.get("question", ""))
        stem, opt_block = split_question_and_options(q_full)
        user_content = prompt_template.format(question=stem, option=opt_block)

        messages: list[dict[str, str]] = []
        if args.system.strip():
            messages.append({"role": "system", "content": args.system.strip()})
        messages.append({"role": "user", "content": user_content})

        gold_answer = str(row.get("gold_answer", ""))
        gold_option = str(row.get("gold_option", ""))

        t_total0 = time.perf_counter()
        responses: list[dict] = []
        total_tokens = 0

        for _ in range(args.n):
            one = run_one_simple_attempt(
                client,
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                top_logprobs=top_lp,
            )
            total_tokens += int(one.get("total_token_count", 0))
            responses.append(one)

        total_time = time.perf_counter() - t_total0

        best_i = min(
            range(len(responses)),
            key=lambda i: _uncertainty_sort_key(responses[i]),
        )
        sel = responses[best_i]
        parsed = sel.get("parsed_option", "")
        correct = is_correct_answer(parsed, gold_option, gold_answer)

        out = {
            "question": q_full,
            "gold_answer": gold_answer,
            "gold_option": gold_option,
            "responses": responses,
            "selected_response_index": best_i,
            "is_selected_correct": correct,
            "total_time_seconds": total_time,
            "total_tokens": total_tokens,
        }

        out_path = out_dir / f"result_{test_idx}.json"
        out_path.write_text(
            json.dumps(out, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        print(f"Wrote {out_path}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
