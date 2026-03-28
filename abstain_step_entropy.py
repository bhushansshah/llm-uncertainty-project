"""
Step-entropy abstention: chunk thinking tokens, compare per-step mean entropy to validation-derived thresholds.

See scripts/abstain_step_entropy_experiment.py for the CLI and README for metric definitions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from utils import compute_entropy

# DeepSeek / Qwen-style think markers (substring match on token text).
# Both must be distinct: previously OPEN was mistakenly the same as CLOSE.
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def get_data_point(d: dict[str, Any]) -> dict[str, Any]:
    """Extract tokens and top_logprobs from a result JSON object."""
    lp = d["response"]["logprobs"]
    return {
        "tokens": lp["tokens"],
        "top_logprobs": lp["top_logprobs"],
    }


def new_token_entropy(token_logprobs_dict: dict[str, float]) -> float:
    """Entropy from a single position's top-k logprob dict (OpenAI/vLLM shape)."""
    token_logprobs = list(token_logprobs_dict.values())
    ps = np.exp(token_logprobs)
    m = ps.sum()
    r = max(0.0, 1.0 - m)
    h = compute_entropy(ps)
    if r > 0:
        h -= r * np.log(r)
    return float(h)


def _find_think_boundaries(tokens: list[str]) -> tuple[int | None, int | None]:
    """Return (start_exclusive, end_exclusive) indices for thinking region, or (None, None)."""
    open_idx: int | None = None
    close_idx: int | None = None
    for i, t in enumerate(tokens):
        if _THINK_OPEN in t:
            if open_idx is None:
                open_idx = i
        if _THINK_CLOSE in t:
            close_idx = i
            break
    return open_idx, close_idx


def slice_thinking_tokens(
    tokens: list[str],
    top_logprobs: list[dict[str, float]],
) -> tuple[list[str], list[dict[str, float]]]:
    """
    Keep only thinking / chain-of-thought tokens.

    - If both opening ``<think>`` and closing ``</think>`` markers appear: slice strictly between them
      (exclusive of marker tokens).
    - If only a closing marker is found: tokens before it (legacy notebook behavior).
    - If no markers: use the full sequence (e.g. instruct models without a separate think block).
    """
    if len(tokens) != len(top_logprobs):
        raise ValueError("tokens and top_logprobs length mismatch")

    open_idx, close_idx = _find_think_boundaries(tokens)

    if open_idx is not None and close_idx is not None and close_idx > open_idx:
        start = open_idx + 1
        end = close_idx
        return tokens[start:end], top_logprobs[start:end]

    if close_idx is not None:
        return tokens[:close_idx], top_logprobs[:close_idx]

    return list(tokens), list(top_logprobs)


def slice_thinking_token_logprobs(
    tokens: list[str],
    top_logprobs: list[dict[str, float]],
    token_logprobs: list[float],
) -> tuple[list[str], list[dict[str, float]], list[float]]:
    """
    Same thinking-region slice as ``slice_thinking_tokens``, but also returns aligned
    ``token_logprobs`` (log p of the sampled token at each position, natural log).
    """
    if len(tokens) != len(top_logprobs) or len(tokens) != len(token_logprobs):
        raise ValueError("tokens, top_logprobs, and token_logprobs length mismatch")

    open_idx, close_idx = _find_think_boundaries(tokens)

    if open_idx is not None and close_idx is not None and close_idx > open_idx:
        start = open_idx + 1
        end = close_idx
        return (
            tokens[start:end],
            top_logprobs[start:end],
            token_logprobs[start:end],
        )

    if close_idx is not None:
        return tokens[:close_idx], top_logprobs[:close_idx], token_logprobs[:close_idx]

    return list(tokens), list(top_logprobs), list(token_logprobs)


def get_per_token_logprob_list(logprobs_block: dict[str, Any]) -> list[float]:
    """
    List of log p (natural log) for the sampled token at each position.

    Supports both field names inside ``response.logprobs``:
    - ``token_logprobs`` (older dumps)
    - ``logprobs`` (newer API / renamed field; same semantics as ``token_logprobs``)
    """
    if "token_logprobs" in logprobs_block:
        return [float(x) for x in logprobs_block["token_logprobs"]]
    if "logprobs" in logprobs_block:
        raw = logprobs_block["logprobs"]
        if isinstance(raw, list):
            return [float(x) for x in raw]
    raise ValueError(
        "response.logprobs must contain either 'token_logprobs' or 'logprobs' "
        "(per-token log probabilities of the sampled tokens)"
    )


def token_neg_log_probs_thinking_only(d: dict[str, Any]) -> list[float]:
    """
    Per-token negative log probability of the **chosen** token in the thinking region:
    ``-log p`` where ``p`` is from ``response.logprobs.token_logprobs`` or
    ``response.logprobs.logprobs`` (natural log).
    """
    lp = d["response"]["logprobs"]
    tpl = lp["top_logprobs"]
    tlp = get_per_token_logprob_list(lp)
    toks, _, chosen_lp = slice_thinking_token_logprobs(lp["tokens"], tpl, tlp)
    if len(toks) != len(chosen_lp):
        raise ValueError("internal slice length mismatch")
    return [-float(x) for x in chosen_lp]


def token_entropies_thinking_only(d: dict[str, Any]) -> list[float]:
    """Per-token entropies for the thinking region only."""
    dp = get_data_point(d)
    toks, tpl = slice_thinking_tokens(dp["tokens"], dp["top_logprobs"])
    return [new_token_entropy(tpl[i]) for i in range(len(toks))]


def chunk_step_means(entropies: list[float], chunk_size: int) -> list[float]:
    """Non-overlapping chunks; last chunk may be shorter; mean entropy per chunk."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if not entropies:
        return []
    out: list[float] = []
    for i in range(0, len(entropies), chunk_size):
        chunk = entropies[i : i + chunk_size]
        out.append(float(np.mean(chunk)))
    return out


def cumulative_sum_chunk_means(chunk_means: list[float]) -> list[float]:
    """
    Per-step aggregated uncertainty: at step ``t``, sum of chunk means from chunk 0 through ``t``
    (inclusive). Same length as ``chunk_means``.
    """
    out: list[float] = []
    s = 0.0
    for m in chunk_means:
        s += float(m)
        out.append(s)
    return out


def cumulative_positive_chunk_increments(chunk_means: list[float]) -> list[float]:
    """
    Cumulative **positive** jumps between consecutive chunk means (same length as ``chunk_means``).

    Let ``m[j]`` be the mean entropy in chunk ``j``. At step ``j``:

    - ``j == 0``: ``0`` (no prior chunk to compare).
    - ``j >= 1``: ``sum_{i=1}^{j} max(0, m[i] - m[i-1])``.

    This matches the ``test.ipynb`` "positive variation in entropy" construction over chunk
    means (non-overlapping chunks; chunking must match ``chunk_step_means``).
    """
    if not chunk_means:
        return []
    out: list[float] = [0.0]
    for j in range(1, len(chunk_means)):
        d = float(chunk_means[j]) - float(chunk_means[j - 1])
        inc = d if d > 0 else 0.0
        out.append(out[-1] + inc)
    return out


def val_step_mean_entropies(
    val_data: list[dict[str, Any]],
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each step index j, mean entropy over correct vs incorrect validation examples
    (only among examples with at least j+1 steps).
    """
    correct_steps: list[list[float]] = []
    incorrect_steps: list[list[float]] = []
    for d in val_data:
        ent = token_entropies_thinking_only(d)
        steps = chunk_step_means(ent, chunk_size)
        if d.get("is_correct"):
            correct_steps.append(steps)
        else:
            incorrect_steps.append(steps)
    return val_step_mean_entropies_from_step_lists(correct_steps, incorrect_steps)


def val_step_means_and_counts_from_step_lists(
    correct_steps: list[list[float]],
    incorrect_steps: list[list[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per step j: mean step value for correct vs incorrect val examples that have step j,
    plus counts n_corr[j], n_inc[j] of how many such examples there are.
    """
    max_len = 0
    for s in correct_steps + incorrect_steps:
        max_len = max(max_len, len(s))

    mean_corr = np.full(max_len, np.nan, dtype=float)
    mean_inc = np.full(max_len, np.nan, dtype=float)
    n_corr = np.zeros(max_len, dtype=int)
    n_inc = np.zeros(max_len, dtype=int)

    for j in range(max_len):
        cvals = [s[j] for s in correct_steps if j < len(s)]
        ivals = [s[j] for s in incorrect_steps if j < len(s)]
        n_corr[j] = len(cvals)
        n_inc[j] = len(ivals)
        if cvals:
            mean_corr[j] = float(np.mean(cvals))
        if ivals:
            mean_inc[j] = float(np.mean(ivals))

    return mean_corr, mean_inc, n_corr, n_inc


def val_step_mean_entropies_from_step_lists(
    correct_steps: list[list[float]],
    incorrect_steps: list[list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Same as val_step_mean_entropies but from precomputed per-example step mean lists."""
    mc, mi, _, _ = val_step_means_and_counts_from_step_lists(
        correct_steps, incorrect_steps
    )
    return mc, mi


def build_active_and_tau(
    mean_corr: np.ndarray,
    mean_inc: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    active[j] iff incorrect mean > correct mean, gap >= delta, and both means defined.
    tau[j] = midpoint when both defined; else nan.
    """
    active = np.zeros(len(mean_corr), dtype=bool)
    tau = np.full(len(mean_corr), np.nan, dtype=float)
    for j in range(len(mean_corr)):
        mc, mi = mean_corr[j], mean_inc[j]
        if np.isnan(mc) or np.isnan(mi):
            continue
        if mi <= mc:
            continue
        if (mi - mc) < delta:
            continue
        active[j] = True
        tau[j] = (mc + mi) / 2.0
    return active, tau


def build_active_and_tau_with_min_support(
    mean_corr: np.ndarray,
    mean_inc: np.ndarray,
    delta: float,
    n_corr: np.ndarray,
    n_inc: np.ndarray,
    min_support_per_class: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Like ``build_active_and_tau``, but a step is only eligible if at least
    ``min_support_per_class`` correct **and** ``min_support_per_class`` incorrect
    validation responses contribute at that step (``n_corr[j]`` and ``n_inc[j]``).
    """
    active = np.zeros(len(mean_corr), dtype=bool)
    tau = np.full(len(mean_corr), np.nan, dtype=float)
    n = len(mean_corr)
    if len(n_corr) != n or len(n_inc) != n:
        raise ValueError("n_corr and n_inc must match mean_corr length")

    for j in range(n):
        if n_corr[j] < min_support_per_class or n_inc[j] < min_support_per_class:
            continue
        mc, mi = mean_corr[j], mean_inc[j]
        if np.isnan(mc) or np.isnan(mi):
            continue
        if mi <= mc:
            continue
        if (mi - mc) < delta:
            continue
        active[j] = True
        tau[j] = (mc + mi) / 2.0
    return active, tau


def count_exceeding_threshold(
    step_means: list[float],
    active: np.ndarray,
    tau: np.ndarray,
    noise: float,
) -> int:
    """How many active steps have step_means[j] > tau[j] + noise."""
    n = min(len(step_means), len(active))
    c = 0
    for j in range(n):
        if not active[j]:
            continue
        if np.isnan(tau[j]):
            continue
        if step_means[j] > tau[j] + noise:
            c += 1
    return c


def should_abstain(
    step_means: list[float],
    active: np.ndarray,
    tau: np.ndarray,
    noise: float,
    ground_threshold: int,
) -> bool:
    return count_exceeding_threshold(step_means, active, tau, noise) >= ground_threshold


def example_usable_for_entropy_abstention(d: dict[str, Any]) -> bool:
    """
    True if ``response.logprobs`` has aligned ``tokens`` and ``top_logprobs`` lists
    (required for ``token_entropies_thinking_only`` / step-entropy and agg pipelines).
    """
    lp = d.get("response", {}).get("logprobs")
    if not isinstance(lp, dict):
        return False
    t = lp.get("tokens")
    tpl = lp.get("top_logprobs")
    if t is None or tpl is None:
        return False
    if not isinstance(t, list) or not isinstance(tpl, list):
        return False
    return len(t) == len(tpl)


def example_usable_for_neg_logprob_abstention(d: dict[str, Any]) -> bool:
    """
    Stricter than entropy: also need per-token log probs (``token_logprobs`` or ``logprobs``)
    aligned with ``tokens`` for ``token_neg_log_probs_thinking_only``.
    """
    if not example_usable_for_entropy_abstention(d):
        return False
    lp = d["response"]["logprobs"]
    try:
        tlp = get_per_token_logprob_list(lp)
    except (ValueError, TypeError, KeyError):
        return False
    return len(tlp) == len(lp["tokens"])


def filter_usable_examples_entropy(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Drop examples that cannot run entropy-based abstention (missing/invalid logprob tokens).

    Call this **before** ``stratified_val_test_split``. The split then applies to the **usable
    pool only**: validation has exactly ``val_size`` rows (when ``len(data) > val_size`` after
    filtering), and test has ``len(data) - val_size`` rows. Hyperparameters (τ, active steps)
    are fit on validation; test evaluation uses only usable examples — same methodology as when
    every row was usable, with a smaller effective dataset.
    """
    return [d for d in data if example_usable_for_entropy_abstention(d)]


def filter_usable_examples_neg_logprob(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Drop examples that cannot run −log p abstention. Same **before-split** semantics as
    ``filter_usable_examples_entropy`` (see that docstring).
    """
    return [d for d in data if example_usable_for_neg_logprob_abstention(d)]


def abstention_f1(
    abstain_flags: list[bool],
    is_correct: list[bool],
) -> tuple[float, float, float, int, int, int]:
    """
    Precision / recall / F1 for abstaining on incorrect answers.

    TP = abstained & not correct
    FP = abstained & correct
    Precision = TP / (TP+FP), Recall = TP / total_incorrect
    """
    n = len(abstain_flags)
    assert len(is_correct) == n

    tp = fp = 0
    total_incorrect = sum(1 for c in is_correct if not c)

    for a, ok in zip(abstain_flags, is_correct):
        if a and not ok:
            tp += 1
        elif a and ok:
            fp += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / total_incorrect if total_incorrect > 0 else 0.0
    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)

    return prec, rec, f1, tp, fp, total_incorrect


def total_tokens_in_response(d: dict[str, Any]) -> int:
    """Full response length (all tokens in logprobs)."""
    lp = d.get("response", {}).get("logprobs")
    if not isinstance(lp, dict):
        return 0
    t = lp.get("tokens")
    if not isinstance(t, list):
        return 0
    return len(t)
