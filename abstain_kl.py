"""
KL(U || p) token-level uncertainty from top-k logprobs with a uniform tail completion.

Full vocabulary size V is required. With only top-k masses p_i = exp(ell_i), S = sum p_i,
r = max(0, 1 - S), we assume the remainder r is spread uniformly over the (V - k) types
not in the top-k list, so p_tail = r / (V - k) for each. Then

  KL(U || p) = -log(V) - (1/V) * ( sum_i log p_i + (V-k) log(r/(V-k)) ),

which is exact KL for this completed distribution (not necessarily the true model KL if the
tail is not uniform). When ``r <= 0``, the tail term is omitted and only the top-``k`` sum is
used (see plan); this avoids an infinite KL from log(0) on unsupported types.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from abstain_step_entropy import (
    example_usable_for_entropy_abstention,
    get_data_point,
    slice_thinking_tokens,
)


def kl_uniform_to_p_from_top_logprobs(
    token_logprobs_dict: dict[str, float],
    vocab_size: int,
) -> float:
    """
    KL(U || p) where U is uniform over ``vocab_size`` tokens and p completes top-k masses
    with uniform tail over the remaining (V - k) vocabulary slots.
    """
    if vocab_size < 1:
        raise ValueError("vocab_size must be >= 1")
    logs = list(token_logprobs_dict.values())
    k = len(logs)
    if k == 0:
        raise ValueError("empty top_logprobs dict")
    if vocab_size < k:
        raise ValueError(f"vocab_size V={vocab_size} must be >= top-k size k={k}")

    ps = np.exp(np.asarray(logs, dtype=np.float64))
    S = float(np.sum(ps))
    r = max(0.0, 1.0 - S)
    sum_log_top = float(np.sum(np.log(np.maximum(ps, 1e-300))))

    if vocab_size == k:
        if r > 1e-9:
            raise ValueError(
                f"vocab_size V={vocab_size} equals top-k k but remainder mass r={r:.6g} > 0"
            )
        sum_log_total = sum_log_top
    elif r <= 0.0:
        sum_log_total = sum_log_top
    else:
        p_tail = r / (vocab_size - k)
        tail_sum_log = (vocab_size - k) * float(np.log(p_tail))
        sum_log_total = sum_log_top + tail_sum_log

    return float(-np.log(vocab_size) - sum_log_total / vocab_size)


def token_kls_thinking_only(d: dict[str, Any], vocab_size: int) -> list[float]:
    """Per-token KL(U || p) for the thinking region only (same slice as entropy)."""
    dp = get_data_point(d)
    toks, tpl = slice_thinking_tokens(dp["tokens"], dp["top_logprobs"])
    return [kl_uniform_to_p_from_top_logprobs(tpl[i], vocab_size) for i in range(len(toks))]


def example_usable_for_kl_abstention(d: dict[str, Any]) -> bool:
    """Same structural requirements as entropy abstention: aligned tokens and top_logprobs."""
    return example_usable_for_entropy_abstention(d)


def filter_usable_examples_kl(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [d for d in data if example_usable_for_kl_abstention(d)]
