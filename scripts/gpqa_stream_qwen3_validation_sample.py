#!/usr/bin/env python3
"""
Load GPQA (diamond) from Hugging Face, stratified val/test split, sample one validation
question, call an OpenAI-compatible chat API with streaming logprobs, print chunk markers
every 50 tokens, and report the token index of ``</redacted_thinking>``.

Requires a server that supports streaming chat completions with ``logprobs`` (e.g. vLLM).
Install: pip install datasets openai scikit-learn numpy
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from openai import OpenAI
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from abstain_step_entropy import _find_think_boundaries  # noqa: E402

USER_PROMPT_TEMPLATE = """You will be given a question. Answer the question by choosing one of the provided options. At the very end of your output, format your answer \\nAnswer: $ANSWER\\n

The chosen answer should be formatted as the following example: 
Example of the options:
A) <OPTION 1>
B) <OPTION 2>
C) <OPTION 3>
D) <OPTION 4>

Final answer: 

Answer: A) <OPTION 1>

Here is the question and the options:

Question: {question}

Options: {option}
"""

OPTION_KEYS = [
    "Correct Answer",
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
]
SYMBOLS = ["A)", "B)", "C)", "D)"]


def format_gpqa_options(row: dict, rng_seed: int) -> str:
    """Deterministic shuffle of four answers (same row always gets same ordering for fixed seed)."""
    rng = np.random.default_rng(rng_seed)
    order = rng.permutation(4)
    lines = [SYMBOLS[i] + str(row[OPTION_KEYS[order[i]]]).strip() for i in range(4)]
    return "\n".join(lines)


def load_gpqa_stratified_val_test(
    val_fraction: float,
    seed: int,
) -> tuple[object, object]:
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    df = ds.to_pandas()
    domains = df["High-level domain"]
    use_stratify = len(domains.unique()) >= 2 and len(df) >= 4
    if use_stratify:
        try:
            val_df, test_df = train_test_split(
                df,
                test_size=1.0 - val_fraction,
                stratify=domains,
                random_state=seed,
            )
            return val_df, test_df
        except ValueError:
            pass
    print(
        "Warning: stratified split failed or unsupported; using random split without stratify.",
        file=sys.stderr,
    )
    return train_test_split(
        df,
        test_size=1.0 - val_fraction,
        random_state=seed,
    )


def pick_one_validation_row(val_df: object, seed: int) -> tuple[object, int]:
    """Return one row and its integer position in val_df (for RNG)."""
    rng = np.random.default_rng(seed)
    pos = int(rng.integers(0, len(val_df)))
    return val_df.iloc[pos], pos


def think_close_token_index(tokens: list[str]) -> int:
    """Index of the token containing ``</redacted_thinking>``, or -1 if absent."""
    _, close_idx = _find_think_boundaries(tokens)
    return int(close_idx) if close_idx is not None else -1


def main() -> None:
    p = argparse.ArgumentParser(
        description="GPQA val sample + streaming Qwen3 with logprobs (OpenAI-compatible API)"
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.3,
        help="Fraction of GPQA train used as validation (rest is test)",
    )
    p.add_argument("--seed", type=int, default=42, help="Split + validation row selection")
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Model id as seen by the API server",
    )
    p.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible base URL (vLLM: http://localhost:PORT/v1)",
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="API key (local vLLM often uses EMPTY)",
    )
    p.add_argument("--max-tokens", type=int, default=32000)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20, help="Passed via extra_body for vLLM")
    p.add_argument("--top-logprobs", type=int, default=10)
    p.add_argument(
        "--system",
        type=str,
        default="",
        help="Optional system message (empty: user message only)",
    )
    p.add_argument(
        "--verbose-tokens",
        action="store_true",
        help="Print each decoded token as it is collected (can be very long)",
    )
    args = p.parse_args()

    vf = args.val_fraction
    if not (0.0 < vf < 1.0):
        p.error("--val-fraction must be strictly between 0 and 1")

    val_df, test_df = load_gpqa_stratified_val_test(vf, args.seed)
    print(
        f"GPQA diamond: train split size {len(val_df) + len(test_df)}; "
        f"validation={len(val_df)}, test={len(test_df)} (val_fraction={vf}, seed={args.seed})",
        flush=True,
    )

    row, pos_in_val = pick_one_validation_row(val_df, args.seed)
    row_idx = int(row.name)
    rng_seed = 42 + row_idx
    options_block = format_gpqa_options(row.to_dict(), rng_seed)
    question_text = str(row["Question"]).strip()
    user_content = USER_PROMPT_TEMPLATE.format(question=question_text, option=options_block)

    print(
        f"Sampled validation row: pandas index={row_idx}, position_in_val={pos_in_val}, "
        f"Record ID={row.get('Record ID', 'n/a')}",
        flush=True,
    )

    messages: list[dict[str, str]] = []
    if args.system.strip():
        messages.append({"role": "system", "content": args.system.strip()})
    messages.append({"role": "user", "content": user_content})

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    print("Streaming completion (logprobs + top_logprobs)...", flush=True)
    tokens: list[str] = []
    token_logprobs: list[float] = []
    top_dicts: list[dict] = []

    stream = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=True,
        logprobs=True,
        top_logprobs=args.top_logprobs,
        extra_body={"top_k": args.top_k},
    )

    n = 0
    for chunk in stream:
        if not chunk.choices:
            continue
        ch = chunk.choices[0]
        lp = ch.logprobs
        if lp is None or lp.content is None:
            continue
        for item in lp.content:
            tokens.append(item.token)
            token_logprobs.append(float(item.logprob))
            td: dict[str, float] = {}
            if item.top_logprobs:
                for alt in item.top_logprobs:
                    td[alt.token] = float(alt.logprob)
            top_dicts.append(td)
            n += 1
            if args.verbose_tokens:
                print(item.token, end="", flush=True)
            if n > 0 and n % 50 == 0:
                print(f"Received chunk {n // 50}", flush=True)

    if args.verbose_tokens:
        print(flush=True)

    if not tokens:
        raise SystemExit(
            "No tokens with logprobs received. Ensure the server supports streaming "
            "chat completions with per-token logprobs (e.g. vLLM with OpenAI chat API)."
        )

    print(f"Total generated tokens: {len(tokens)}", flush=True)
    print(f"think_end_token_index (token containing </redacted_thinking>): {think_close_token_index(tokens)}", flush=True)

    # Sanity: first token logprob
    if token_logprobs:
        print(f"First token logprob: {token_logprobs[0]:.6f}", flush=True)
    if top_dicts:
        print(f"First position top_logprobs keys (sample): {list(top_dicts[0].keys())[:5]}...", flush=True)


if __name__ == "__main__":
    main()
