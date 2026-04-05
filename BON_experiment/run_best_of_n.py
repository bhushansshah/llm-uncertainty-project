#!/usr/bin/env python3
"""
Best-of-N with streaming step-entropy abstention (GPQA + calibration from ``result_*.json``).

**Calibration contract**

- ``--calibration-results-dir`` must contain ``result_*.json`` files loadable by
  ``abstain_experiment_common.load_results_flat_dir`` (same shape as existing GPQA runs: ``question``,
  ``gold_answer``, ``gold_option``, ``response.logprobs.tokens`` / ``top_logprobs``, ``is_correct``).
- Examples are filtered with ``filter_usable_examples_entropy``; then split with
  ``stratified_val_test_split_fraction`` (same seed/fraction as when you produced these files, if you
  want τ to match offline scripts).
- **Test** rows reuse ``question`` / gold fields from the JSON; stored ``response`` is ignored for
  new generations.

**Server**

- Use an OpenAI-compatible endpoint that streams **logprobs for every token** (e.g. vLLM). If many
  chunks omit logprobs, abstention will not match offline step-entropy behavior.
- Hugging Face Inference Providers (``router.huggingface.co``) do **not** support ``extra_body.top_k``;
  keep ``--top-k 0`` (default). Pass ``--top-k 20`` (or similar) only for servers like vLLM that accept it.

**Last-attempt rule**

- Attempts ``1 .. N-1`` use abstention. Attempt ``N`` runs **without** abstention iff every prior
  attempt abstained (for ``N == 1``, the single attempt never abstains).

**Outputs**

- Shared hyperparameters and τ/active are written once to ``<output_root>/<dataset>/<model>/config.json``.
- Per-question results are ``result_{idx}.json`` in that same directory (no duplicated config).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
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
from abstain_step_entropy import (  # noqa: E402
    build_active_and_tau_with_min_support,
    chunk_step_means,
    filter_usable_examples_entropy,
    token_entropies_thinking_only,
    val_step_means_and_counts_from_step_lists,
)
from generate_responses_verbalized import extract_answer, is_correct_answer  # noqa: E402

from BON_experiment.streaming_abstain import (  # noqa: E402
    stream_chat_with_optional_abstention,
)

DEFAULT_PROMPT = """You will be given a question. Answer the question by choosing one of the provided options. At the very end of your output, format your answer \\nAnswer: $ANSWER\\n

The chosen answer should be formatted as the following example: 
Example of the options:
A)<OPTION 1>
B)<OPTION 2>
C)<OPTION 3>
D)<OPTION 4>

Final answer: 

Answer: A)<OPTION 1>

Here is the question and the options:

Question: {question}

Options: {option}
"""


def val_step_entropy_stats(
    val_data: list[dict],
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    correct_steps: list[list[float]] = []
    incorrect_steps: list[list[float]] = []
    for d in val_data:
        steps = chunk_step_means(token_entropies_thinking_only(d), chunk_size)
        if d.get("is_correct"):
            correct_steps.append(steps)
        else:
            incorrect_steps.append(steps)
    return val_step_means_and_counts_from_step_lists(correct_steps, incorrect_steps)


def split_question_and_options(stored_question: str) -> tuple[str, str]:
    """Split a stored GPQA ``question`` field into stem + options block for ``{question}`` / ``{option}``."""
    if "\nA)" in stored_question:
        stem, rest = stored_question.split("\nA)", 1)
        return stem.strip(), "A)" + rest.strip()
    return stored_question.strip(), ""


def _json_default(o: object):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _uncertainty_sort_key(r: dict) -> float:
    v = r.get("uncertainty")
    if v is None:
        return float("inf")
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("inf")
    if np.isnan(x):
        return float("inf")
    return x


def letter_from_parsed(parsed: str) -> str:
    s = (parsed or "").strip()
    if not s:
        return ""
    c = s[0].upper()
    if c in "ABCD":
        return c
    return s


def build_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def run_one_attempt(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    top_logprobs: int,
    active: np.ndarray,
    tau: np.ndarray,
    offset: float,
    ground_threshold: int,
    chunk_size: int,
    abstain_enabled: bool,
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

    raw = stream_chat_with_optional_abstention(
        stream,
        active=active,
        tau=tau,
        offset=offset,
        ground_threshold=ground_threshold,
        chunk_size=chunk_size,
        abstain_enabled=abstain_enabled,
    )

    full_text = "".join(raw["tokens"])
    if raw["is_abstained"]:
        raw["parsed_option"] = ""
    else:
        parsed = extract_answer(full_text)
        raw["parsed_option"] = letter_from_parsed(parsed)
    return raw


def main() -> None:
    p = argparse.ArgumentParser(description="Best-of-N with streaming step-entropy abstention")
    p.add_argument(
        "--calibration-results-dir",
        type=str,
        required=True,
        help="Directory with result_*.json (usable for entropy abstention)",
    )
    p.add_argument("--val-fraction", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-size", type=int, required=True)
    p.add_argument("--delta", type=float, required=True)
    p.add_argument("--offset", type=float, required=True)
    p.add_argument("--ground-threshold", type=int, required=True)
    p.add_argument("--min-support", type=int, default=3, dest="min_support_per_class")
    p.add_argument("--n", type=int, default=4, help="Best-of-N attempts")
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
        help="If >0, passed as extra_body top_k (vLLM). Use 0 for Hugging Face Inference Providers "
        "(they reject top_k).",
    )
    p.add_argument("--top-logprobs", type=int, default=10)
    p.add_argument(
        "--prompt",
        type=str,
        default="",
        help="User prompt template with {question} and {option}; default: GPQA template",
    )
    p.add_argument(
        "--dataset-name",
        type=str,
        default="gpqa",
        help="Subfolder under best_of_N_outputs",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default=str(_ROOT / "best_of_N_outputs"),
        help="Root directory for outputs",
    )
    p.add_argument("--test-limit", type=int, default=0, help="If >0, only first K test questions")
    p.add_argument("--system", type=str, default="", help="Optional system message")
    args = p.parse_args()

    if not (0.0 < args.val_fraction < 1.0):
        p.error("--val-fraction must be between 0 and 1")
    if args.n < 1:
        p.error("--n must be >= 1")
    if args.chunk_size < 1:
        p.error("--chunk-size must be >= 1")

    prompt_template = args.prompt.strip() or DEFAULT_PROMPT

    raw_data = load_results_flat_dir(args.calibration_results_dir)
    usable = filter_usable_examples_entropy(raw_data)
    if len(usable) < 2:
        raise SystemExit(f"Need >= 2 usable examples after filtering, got {len(usable)}")

    val_data, test_data = stratified_val_test_split_fraction(
        usable, args.val_fraction, args.seed
    )

    mean_corr, mean_inc, n_corr, n_inc = val_step_entropy_stats(val_data, args.chunk_size)
    active, tau = build_active_and_tau_with_min_support(
        mean_corr,
        mean_inc,
        float(args.delta),
        n_corr,
        n_inc,
        int(args.min_support_per_class),
    )

    top_lp = max(1, min(int(args.top_logprobs), 20))

    client = build_client(args.base_url, args.api_key)
    model_safe = sanitize_filename_component(args.model)
    ds_safe = sanitize_filename_component(args.dataset_name)
    out_dir = Path(args.output_root) / ds_safe / model_safe
    out_dir.mkdir(parents=True, exist_ok=True)

    config_block = {
        "calibration_results_dir": os.path.abspath(args.calibration_results_dir),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "chunk_size": args.chunk_size,
        "delta": args.delta,
        "offset": args.offset,
        "ground_threshold": args.ground_threshold,
        "min_support_per_class": args.min_support_per_class,
        "n": args.n,
        "model": args.model,
        "base_url": args.base_url,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "top_logprobs": top_lp,
        "dataset_name": args.dataset_name,
        "tau": [float(x) if np.isfinite(x) else None for x in tau],
        "active": [bool(x) for x in active],
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

        for k in range(args.n):
            if k < args.n - 1:
                abstain_en = True
            else:
                if args.n == 1:
                    abstain_en = False
                else:
                    all_prior_abstained = all(r.get("is_abstained") for r in responses)
                    abstain_en = not all_prior_abstained

            one = run_one_attempt(
                client,
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                top_logprobs=top_lp,
                active=active,
                tau=tau,
                offset=float(args.offset),
                ground_threshold=int(args.ground_threshold),
                chunk_size=int(args.chunk_size),
                abstain_enabled=abstain_en,
            )
            total_tokens += int(one.get("total_token_count", 0))
            responses.append(one)

        total_time = time.perf_counter() - t_total0

        non_abs = [i for i, r in enumerate(responses) if not r.get("is_abstained")]
        if non_abs:
            best_i = min(non_abs, key=lambda i: _uncertainty_sort_key(responses[i]))
        else:
            best_i = min(range(len(responses)), key=lambda i: _uncertainty_sort_key(responses[i]))

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
