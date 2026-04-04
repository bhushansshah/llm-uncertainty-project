#!/usr/bin/env python3
"""
Hugging Face Inference Router: stream a chat completion, parse logprob chunks, save ``tmp.json``.

Set ``HF_TOKEN`` in the project root ``.env``. Optional env: ``HF_ROUTER_BASE``,
``HF_ROUTER_MODEL``, ``TOP_LOGPROBS`` (also accepts legacy ``TEST_*`` names). Defaults: base
``https://router.huggingface.co/v1``, model ``Qwen/Qwen3-32B:groq``.

The router often sends (1) chunks with ``choices[0].logprobs.content`` and (2) later chunks with
text in ``delta.content`` only. This script **keeps only** ``logprobs.content`` items that include a
non-null ``logprob`` (same idea as OpenAI streaming logprobs).

Docs: https://huggingface.co/docs/inference-providers/tasks/chat-completion

Run:  python tests/test_streaming_logprobs_providers.py

Requires: pip install openai python-dotenv
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
OUT_JSON = _REPO / "tmp.json"

try:
    from dotenv import load_dotenv
except ImportError as e:
    raise SystemExit("Install python-dotenv: pip install python-dotenv") from e

from openai import OpenAI

load_dotenv(_REPO / ".env")

PROMPT = "What is the capital of India? Answer in one sentence only."
BASE_URL = os.environ.get("HF_ROUTER_BASE") or os.environ.get(
    "TEST_HF_ROUTER_BASE", "https://router.huggingface.co/v1"
)
MODEL = os.environ.get("HF_ROUTER_MODEL") or os.environ.get(
    "TEST_HF_ROUTER_MODEL", "Qwen/Qwen3-32B:groq"
)
TOP_LOGPROBS = min(
    int(os.environ.get("TOP_LOGPROBS") or os.environ.get("TEST_TOP_LOGPROBS", "5")), 5
)


def parse_stream(stream) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Walk SSE chunks. For each chunk, if ``logprobs.content`` is present, append one row per token
    that has a ``logprob``. Skip chunks that only carry ``delta.content`` (no logprobs).
    """
    rows: list[dict[str, Any]] = []
    n_chunks = 0
    n_text_only_chunks = 0

    for chunk in stream:
        n_chunks += 1
        if not chunk.choices:
            continue
        ch = chunk.choices[0]
        lp = ch.logprobs
        emitted = False

        if lp is not None and lp.content:
            for item in lp.content:
                chosen = getattr(item, "logprob", None)
                if chosen is None:
                    continue
                tops: list[dict[str, Any]] = []
                for t in item.top_logprobs or []:
                    tl = getattr(t, "logprob", None)
                    tops.append(
                        {"token": t.token, "logprob": float(tl) if tl is not None else None}
                    )
                rows.append(
                    {
                        "token": item.token,
                        "logprob": float(chosen),
                        "top_logprobs": tops,
                    }
                )
                emitted = True

        delta = ch.delta
        text = getattr(delta, "content", None) if delta else None
        if text and not emitted:
            n_text_only_chunks += 1

    summary = {
        "chunks_seen": n_chunks,
        "text_only_chunks": n_text_only_chunks,
        "token_rows": len(rows),
    }
    return rows, summary


def main() -> None:
    key = os.environ.get("HF_TOKEN", "").strip()
    if not key:
        print("Set HF_TOKEN in .env", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=key, base_url=BASE_URL)
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=512,
        temperature=0,
        stream=True,
        logprobs=True,
        top_logprobs=TOP_LOGPROBS,
    )

    tokens, summary = parse_stream(stream)
    out: dict[str, Any] = {
        "meta": {
            "base_url": BASE_URL,
            "model": MODEL,
            "top_logprobs": TOP_LOGPROBS,
            "prompt": PROMPT,
        },
        "summary": summary,
        "tokens": tokens,
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {summary['token_rows']} rows to {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
