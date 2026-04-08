"""
Stream an OpenAI-compatible chat completion, compute step-entropy abstention on the thinking
region, and optionally stop early when ``should_abstain`` fires.

Requires a server that returns **logprobs for every streamed token** (e.g. vLLM). If some chunks
omit logprobs, token lists will be incomplete and abstention may be unreliable.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from abstain_step_entropy import (
    _find_think_boundaries,
    chunk_step_means,
    new_token_entropy,
    should_abstain,
)

# Same closing marker as ``abstain_step_entropy`` (must match for split-token safety).
THINK_CLOSE_TAG = "</think>"


def _top_alternatives_to_dict(alternatives: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if not alternatives:
        return out
    for alt in alternatives:
        tok = getattr(alt, "token", None)
        lp = getattr(alt, "logprob", None)
        if tok is not None and lp is not None:
            out[str(tok)] = float(lp)
    return out


def thinking_entropies_prefix(
    tokens: list[str],
    top_logprobs: list[dict[str, float]],
) -> list[float]:
    """
    Per-token entropies for the **thinking** region, consistent with streaming:

    - If both think markers exist: strictly between them (same as ``slice_thinking_tokens``).
    - If only ``<think>`` is seen so far: tokens after the open marker through end
      (prefix of the eventual thinking block).
    - If only close exists (unusual): tokens before close.
    - If neither marker is found: full sequence ``[0:n]`` (same as offline ``slice_thinking_tokens``
      when no think tags are present).
    """
    if len(tokens) != len(top_logprobs):
        raise ValueError("tokens and top_logprobs length mismatch")

    open_idx, close_idx = _find_think_boundaries(tokens)
    n = len(tokens)

    if open_idx is not None and close_idx is not None and close_idx > open_idx:
        start, end = open_idx + 1, close_idx
    elif close_idx is not None:
        start, end = 0, close_idx
    elif open_idx is not None:
        start, end = open_idx + 1, n
    else:
        start, end = 0, n

    ent: list[float] = []
    for i in range(start, end):
        ent.append(new_token_entropy(top_logprobs[i]))
    return ent


def full_text_from_tokens(tokens: list[str]) -> str:
    return "".join(tokens)


def split_reasoning_and_content(tokens: list[str]) -> tuple[str, str]:
    """Join tokens; split into reasoning (think block) and tail content after ``</redacted_thinking>``."""
    text = full_text_from_tokens(tokens)
    close = "</think>"
    if close in text:
        pre, _, post = text.partition(close)
        return pre + close, post
    return text, ""


def mean_thinking_entropy(tokens: list[str], top_logprobs: list[dict[str, float]]) -> float:
    ent = thinking_entropies_prefix(tokens, top_logprobs)
    if not ent:
        return float("nan")
    return float(np.mean(ent))


def stream_chat_collect_only(stream) -> dict[str, Any]:
    """
    Consume the full streaming iterator: collect all tokens with logprobs until the stream ends.
    No early abstention. ``uncertainty`` is mean token entropy over the thinking region (same as
    ``stream_chat_with_optional_abstention`` when it does not abstain).
    """
    t0 = time.perf_counter()
    tokens: list[str] = []
    log_probs: list[float] = []
    top_logprobs: list[dict[str, float]] = []
    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            ch = chunk.choices[0]
            lp = ch.logprobs
            if lp is None or lp.content is None:
                continue
            for item in lp.content:
                chosen = getattr(item, "logprob", None)
                if chosen is None:
                    continue
                tokens.append(item.token)
                log_probs.append(float(chosen))
                top_logprobs.append(_top_alternatives_to_dict(item.top_logprobs))
    finally:
        close_fn = getattr(stream, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

    elapsed = time.perf_counter() - t0
    reasoning, content = split_reasoning_and_content(tokens)
    unc = mean_thinking_entropy(tokens, top_logprobs)

    return {
        "tokens": tokens,
        "log_probs": log_probs,
        "top_logprobs": top_logprobs,
        "is_abstained": False,
        "reasoning": reasoning,
        "content": content,
        "uncertainty": unc,
        "total_token_count": len(tokens),
        "time_taken": elapsed,
    }


def stream_chat_with_optional_abstention(
    stream,
    *,
    active: np.ndarray,
    tau: np.ndarray,
    offset: float,
    ground_threshold: int,
    chunk_size: int,
    abstain_enabled: bool,
) -> dict[str, Any]:
    """
    Consume an OpenAI streaming iterator; collect tokens and logprobs.

    If ``abstain_enabled``, after each **completed** non-overlapping chunk of **thinking**
    entropies, evaluate ``should_abstain`` on the prefix of chunk-step means; stop the stream
    early if true. Abstention checks stop once the closing think tag has appeared in the
    concatenated stream (so we never abstain based on tokens **after** the thinking block).
    """
    t0 = time.perf_counter()
    tokens: list[str] = []
    log_probs: list[float] = []
    top_logprobs: list[dict[str, float]] = []
    is_abstained = False
    think_close_found = False
    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            ch = chunk.choices[0]
            lp = ch.logprobs
            if lp is None or lp.content is None:
                continue
            for item in lp.content:
                chosen = getattr(item, "logprob", None)
                if chosen is None:
                    continue
                tokens.append(item.token)
                if not think_close_found and THINK_CLOSE_TAG in "".join(tokens):
                    think_close_found = True
                log_probs.append(float(chosen))
                top_logprobs.append(_top_alternatives_to_dict(item.top_logprobs))

                if not abstain_enabled or think_close_found:
                    continue

                ent_series = thinking_entropies_prefix(tokens, top_logprobs)
                if len(ent_series) < chunk_size or len(ent_series) % chunk_size != 0:
                    continue
                step_means = chunk_step_means(ent_series, chunk_size)
                if should_abstain(
                    step_means,
                    active,
                    tau,
                    offset,
                    ground_threshold,
                ):
                    is_abstained = True
                    break
            if is_abstained:
                break
    finally:
        close_fn = getattr(stream, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

    elapsed = time.perf_counter() - t0
    reasoning, content = split_reasoning_and_content(tokens)
    unc = mean_thinking_entropy(tokens, top_logprobs)

    return {
        "tokens": tokens,
        "log_probs": log_probs,
        "top_logprobs": top_logprobs,
        "is_abstained": is_abstained,
        "reasoning": reasoning,
        "content": content,
        "uncertainty": unc,
        "total_token_count": len(tokens),
        "time_taken": elapsed,
    }
