"""Canonical token-usage accounting seam shared by all provider clients.

Each provider SDK reports token usage in its own idiosyncratic shape. This
module collapses that heterogeneity into a single intermediate representation,
``NormalizedUsage``, and a small set of provider-specific adapter functions that
map a raw provider usage payload onto it. ``to_token_usage`` then converts a
``NormalizedUsage`` into the public ``TokenUsage`` contract, passing every count
through unchanged so the existing schema-level auto-fill stays a no-op when the
counts are already known.

The split between reasoning and visible-answer tokens is always repaired to
stay non-negative and to preserve ``answer_tokens + reasoning_tokens ==
output_tokens``. Some OpenAI-compatible relays report ``reasoning_tokens``
larger than their combined output count; in that case the provider's
``output_tokens`` total is kept, visible answer tokens are estimated from the
final text when available, and the remainder is assigned to reasoning.
Whenever the answer share comes from an estimate rather than an exact provider
count (a repaired impossible split, or the Anthropic count_tokens fallback),
``answer_tokens_estimated`` is set so rows with estimated splits stay
distinguishable in the cache and results.

The Anthropic path uses the provider's exact ``thinking_tokens`` detail when
the stream reports it; only relays that omit the field fall back to deriving
the answer count from the visible text via a tokenizer. That fallback needs a
network call, so it lives behind an injected ``count_tokens`` callback and an
``async`` normalize function, which keeps this module free of I/O and
unit-testable without a network.
"""
import logging
import math
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Optional, Tuple

from ..core.schemas import TokenUsage

logger = logging.getLogger(__name__)


def _estimate_visible_tokens(final_text: str) -> int:
    """Cheap fallback used only when a provider reports an impossible split."""
    if not final_text:
        return 0
    return max(1, math.ceil(len(final_text) / 4))


def _fold_hidden_total_gap(
    prompt_tokens: int, output_tokens: int, reasoning_tokens: int, total_tokens: int,
) -> tuple[int, int, int]:
    """Return ``(output_tokens, reasoning_tokens, unattributed_tokens)`` with any
    hidden-total gap folded in.

    Some relays bill hidden thinking only in the total: prompt and output cover
    the visible exchange while ``total - prompt - output`` is a positive
    remainder (observed: relayed gemini-3.5-flash streaming, 2.7k-3.9k per
    row). In this single-turn contract the total can only be prompt plus
    output, so a positive gap is hidden output: fold it into output/reasoning
    and report it as ``unattributed_tokens`` so the provider's original
    attribution stays auditable. A missing total or a negative gap leaves the
    counts provider-literal.
    """
    gap = total_tokens - prompt_tokens - output_tokens
    if gap <= 0:
        return output_tokens, reasoning_tokens, 0
    return output_tokens + gap, reasoning_tokens + gap, gap


def _repair_output_split(output_tokens: int, reasoning_tokens: int, final_text: str = "") -> tuple[int, int, bool]:
    """Return ``(answer_tokens, reasoning_tokens, answer_estimated)`` with no
    negative counts.

    The provider's combined output count is treated as authoritative. If the
    reported reasoning count fits inside it, keep the provider split. If it does
    not, estimate the visible-answer share from text and assign the remainder to
    reasoning, preserving the combined output count; ``answer_estimated`` is
    True exactly when that estimate path was taken.
    """
    output_tokens = max(int(output_tokens or 0), 0)
    reasoning_tokens = max(int(reasoning_tokens or 0), 0)

    if reasoning_tokens <= output_tokens:
        return output_tokens - reasoning_tokens, reasoning_tokens, False

    answer_tokens = min(_estimate_visible_tokens(final_text), output_tokens)
    return answer_tokens, output_tokens - answer_tokens, True


@dataclass
class NormalizedUsage:
    """Provider-agnostic token counts plus optional cache diagnostics.

    The first five fields are the canonical accounting matrix and map one-to-one
    onto ``TokenUsage``. The cache fields mirror the names Anthropic reports; they
    are carried for logging/diagnostics only and are not surfaced in the public
    token schema.
    """

    prompt_tokens: int = 0
    reasoning_tokens: int = 0
    answer_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # True when the answer/reasoning split rests on a tokenizer estimate
    # instead of an exact provider count; carried into TokenUsage so estimated
    # splits stay auditable per row.
    answer_tokens_estimated: bool = False
    # Tokens present in the provider's total but attributed to neither prompt
    # nor completion by the provider (hidden thinking billed only in the
    # total). They are folded into output/reasoning; this count preserves the
    # provider's original attribution for audit.
    unattributed_tokens: int = 0
    # Cache diagnostics (Anthropic). Logged only; intentionally not propagated
    # into TokenUsage / ExperimentSummary.
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    cache_ephemeral_5m: int = 0
    cache_ephemeral_1h: int = 0


def to_token_usage(usage: NormalizedUsage) -> TokenUsage:
    """Convert a ``NormalizedUsage`` into the public ``TokenUsage`` contract.

    Every count is passed through explicitly; ``TokenUsage.__init__`` only fills
    ``output``/``total`` when they are left at zero, so a fully-populated
    ``NormalizedUsage`` round-trips unchanged. Cache diagnostics are deliberately
    dropped here.
    """
    return TokenUsage(
        prompt_tokens=usage.prompt_tokens,
        answer_tokens=usage.answer_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        answer_tokens_estimated=usage.answer_tokens_estimated,
        unattributed_tokens=usage.unattributed_tokens,
    )


def extract_openai_usage(usage: Any, final_text: str = "") -> NormalizedUsage:
    """Map an OpenAI-compatible chat-completions ``usage`` object.

    Reasoning tokens come from ``completion_tokens_details.reasoning_tokens`` when
    present (else zero). The combined completion count stays authoritative; an
    impossible split is repaired using ``final_text`` when available, and a
    hidden-total gap is folded in per ``_fold_hidden_total_gap``.
    """
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or 0

    reasoning_tokens = 0
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

    output_tokens, reasoning_tokens, unattributed_tokens = _fold_hidden_total_gap(
        prompt_tokens, completion_tokens, reasoning_tokens, total_tokens,
    )

    answer_tokens, reasoning_tokens, answer_estimated = _repair_output_split(
        output_tokens, reasoning_tokens, final_text,
    )

    return NormalizedUsage(
        prompt_tokens=prompt_tokens,
        answer_tokens=answer_tokens,
        reasoning_tokens=reasoning_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        answer_tokens_estimated=answer_estimated,
        unattributed_tokens=unattributed_tokens,
    )


def extract_responses_usage(usage: Optional[Mapping[str, Any]], final_text: str = "") -> NormalizedUsage:
    """Map an OpenAI Responses API ``usage`` dict from the completed event.

    Reasoning tokens come from ``output_tokens_details.reasoning_tokens`` when
    present (else zero). The combined output count stays authoritative; an
    impossible split is repaired using ``final_text`` when available, and a
    hidden-total gap is folded in per ``_fold_hidden_total_gap``.
    """
    usage = usage or {}
    prompt_tokens = usage.get("input_tokens", 0) or 0
    output_tokens = usage.get("output_tokens", 0) or 0
    total_tokens = usage.get("total_tokens", 0) or 0

    reasoning_tokens = 0
    details = usage.get("output_tokens_details") or {}
    if details:
        reasoning_tokens = details.get("reasoning_tokens", 0) or 0

    output_tokens, reasoning_tokens, unattributed_tokens = _fold_hidden_total_gap(
        prompt_tokens, output_tokens, reasoning_tokens, total_tokens,
    )

    answer_tokens, reasoning_tokens, answer_estimated = _repair_output_split(
        output_tokens, reasoning_tokens, final_text,
    )

    return NormalizedUsage(
        prompt_tokens=prompt_tokens,
        answer_tokens=answer_tokens,
        reasoning_tokens=reasoning_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        answer_tokens_estimated=answer_estimated,
        unattributed_tokens=unattributed_tokens,
    )


def extract_gemini_usage(metadata: Any) -> NormalizedUsage:
    """Map Gemini ``usage_metadata`` through its documented fallback chains.

    The primary field name wins; a fallback name is consulted only when the
    primary value is falsy. Output tokens are reasoning plus answer (Gemini does
    not report a combined output count), and the total falls back to the sum when
    not reported directly. A reported total larger than prompt plus output is
    folded in per ``_fold_hidden_total_gap`` (Gemini's total includes thoughts,
    so a relay that omits ``thoughts_token_count`` leaves exactly this gap).
    ``None`` metadata yields an all-zero result.
    """
    if metadata is None:
        return NormalizedUsage()

    prompt_tokens = getattr(metadata, "prompt_token_count", None) or 0
    answer_tokens = getattr(metadata, "candidates_token_count", None) or 0
    total_tokens = getattr(metadata, "total_token_count", None) or 0

    reasoning_tokens = int(getattr(metadata, "thoughts_token_count", None) or 0)

    if not prompt_tokens:
        prompt_tokens = getattr(metadata, "prompt_tokens", None) or 0
    if not answer_tokens:
        answer_tokens = getattr(metadata, "completion_tokens", None) or 0
    if not total_tokens:
        total_tokens = getattr(metadata, "total_tokens", None) or 0

    if not prompt_tokens:
        prompt_tokens = getattr(metadata, "input_tokens", None) or 0
    if not answer_tokens:
        answer_tokens = getattr(metadata, "output_tokens", None) or 0

    prompt_tokens = int(prompt_tokens) if prompt_tokens else 0
    answer_tokens = int(answer_tokens) if answer_tokens else 0
    total_tokens = int(total_tokens) if total_tokens else 0

    output_tokens = reasoning_tokens + answer_tokens
    if not total_tokens:
        total_tokens = prompt_tokens + answer_tokens + reasoning_tokens

    output_tokens, reasoning_tokens, unattributed_tokens = _fold_hidden_total_gap(
        prompt_tokens, output_tokens, reasoning_tokens, total_tokens,
    )

    return NormalizedUsage(
        prompt_tokens=prompt_tokens,
        answer_tokens=answer_tokens,
        reasoning_tokens=reasoning_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        unattributed_tokens=unattributed_tokens,
    )


async def normalize_anthropic_usage(
    *,
    prompt_tokens: int,
    output_tokens: int,
    final_text: str,
    count_tokens: Callable[[str], Awaitable[Tuple[int, bool]]],
    thinking_tokens: Optional[int] = None,
    cache_metrics: Optional[Mapping[str, int]] = None,
) -> NormalizedUsage:
    """Normalize Anthropic usage, preferring the provider's exact thinking count.

    The official API reports ``output_tokens_details.thinking_tokens`` in the
    final ``message_delta``; when the caller passes it and it fits inside the
    combined output, the split is taken verbatim (reasoning = thinking, answer =
    remainder) with no extra request. Only when the field is absent (compatible
    relays) or impossible is the answer count derived by running the visible
    answer text through the injected ``count_tokens`` callback (which owns the
    count-tokens HTTP call and its tokenizer fallback, and reports ``(count,
    estimated)`` so estimate-based splits stay auditable). Empty text skips the
    callback entirely. The derived answer count is clamped to the reported
    output, and reasoning is the remainder.

    Cache diagnostics, when present, are carried on the result and logged here so
    cache accounting stays observable without being surfaced in ``TokenUsage``.
    """
    if thinking_tokens is not None and 0 <= thinking_tokens <= output_tokens:
        answer_tokens = output_tokens - thinking_tokens
        answer_estimated = False
    elif final_text:
        counted, answer_estimated = await count_tokens(final_text)
        answer_tokens = min(counted, output_tokens)
    else:
        answer_tokens, answer_estimated = 0, False
    reasoning_tokens = max(output_tokens - answer_tokens, 0)

    metrics = cache_metrics or {}
    normalized = NormalizedUsage(
        prompt_tokens=prompt_tokens,
        answer_tokens=answer_tokens,
        reasoning_tokens=reasoning_tokens,
        output_tokens=output_tokens,
        total_tokens=prompt_tokens + output_tokens,
        answer_tokens_estimated=answer_estimated,
        cache_creation_tokens=metrics.get("cache_creation_tokens", 0) or 0,
        cache_read_tokens=metrics.get("cache_read_tokens", 0) or 0,
        cache_ephemeral_5m=metrics.get("cache_ephemeral_5m", 0) or 0,
        cache_ephemeral_1h=metrics.get("cache_ephemeral_1h", 0) or 0,
    )

    if normalized.cache_creation_tokens or normalized.cache_read_tokens:
        logger.info(
            f"cache: creation={normalized.cache_creation_tokens} "
            f"(5m={normalized.cache_ephemeral_5m}, 1h={normalized.cache_ephemeral_1h}) "
            f"read={normalized.cache_read_tokens}"
        )

    return normalized
