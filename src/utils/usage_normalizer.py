"""Canonical token-usage accounting seam shared by all provider clients.

Each provider SDK reports token usage in its own idiosyncratic shape. This
module collapses that heterogeneity into a single intermediate representation,
``NormalizedUsage``, and a small set of provider-specific adapter functions that
map a raw provider usage payload onto it. ``to_token_usage`` then converts a
``NormalizedUsage`` into the public ``TokenUsage`` contract, passing every count
through unchanged so the existing schema-level auto-fill stays a no-op when the
counts are already known.

The adapters reproduce each provider's existing arithmetic exactly; no new
clamping or guarding is introduced for the chat-completions, Responses, or
Gemini paths (so a derived answer count may be negative when the provider
reports more reasoning than total output, matching prior behavior). The
Anthropic path is the only one that requires deriving the answer count from the
visible text via a tokenizer; that work needs a network call, so it lives behind
an injected ``count_tokens`` callback and an ``async`` normalize function, which
keeps this module free of I/O and unit-testable without a network.
"""
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Optional

from ..core.schemas import TokenUsage

logger = logging.getLogger(__name__)


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
    )


def extract_openai_usage(usage: Any) -> NormalizedUsage:
    """Map an OpenAI-compatible chat-completions ``usage`` object.

    Reasoning tokens come from ``completion_tokens_details.reasoning_tokens`` when
    present (else zero); the answer count is the remainder of the completion
    tokens. No clamping is applied, matching the chat-completions client.
    """
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    reasoning_tokens = 0
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

    answer_tokens = completion_tokens - reasoning_tokens

    return NormalizedUsage(
        prompt_tokens=prompt_tokens,
        answer_tokens=answer_tokens,
        reasoning_tokens=reasoning_tokens,
        output_tokens=completion_tokens,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
    )


def extract_responses_usage(usage: Optional[Mapping[str, Any]]) -> NormalizedUsage:
    """Map an OpenAI Responses API ``usage`` dict from the completed event.

    Reasoning tokens come from ``output_tokens_details.reasoning_tokens`` when
    present (else zero); the answer count is the remainder of the output tokens.
    """
    usage = usage or {}
    prompt_tokens = usage.get("input_tokens", 0) or 0
    output_tokens = usage.get("output_tokens", 0) or 0
    total_tokens = usage.get("total_tokens", 0) or 0

    reasoning_tokens = 0
    details = usage.get("output_tokens_details") or {}
    if details:
        reasoning_tokens = details.get("reasoning_tokens", 0) or 0

    answer_tokens = output_tokens - reasoning_tokens

    return NormalizedUsage(
        prompt_tokens=prompt_tokens,
        answer_tokens=answer_tokens,
        reasoning_tokens=reasoning_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def extract_gemini_usage(metadata: Any) -> NormalizedUsage:
    """Map Gemini ``usage_metadata`` through its documented fallback chains.

    The primary field name wins; a fallback name is consulted only when the
    primary value is falsy. Output tokens are reasoning plus answer (Gemini does
    not report a combined output count), and the total falls back to the sum when
    not reported directly. ``None`` metadata yields an all-zero result.
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

    return NormalizedUsage(
        prompt_tokens=prompt_tokens,
        answer_tokens=answer_tokens,
        reasoning_tokens=reasoning_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


async def normalize_anthropic_usage(
    *,
    prompt_tokens: int,
    output_tokens: int,
    final_text: str,
    count_tokens: Callable[[str], Awaitable[int]],
    cache_metrics: Optional[Mapping[str, int]] = None,
) -> NormalizedUsage:
    """Normalize Anthropic usage, deriving the answer split via ``count_tokens``.

    Anthropic reports only a combined ``output_tokens`` count, so the answer
    count is obtained by running the visible answer text through the injected
    ``count_tokens`` callback (which owns the count-tokens HTTP call and its
    tokenizer fallback). Empty text skips the callback entirely. The answer count
    is clamped to the reported output, and reasoning is the remainder.

    Cache diagnostics, when present, are carried on the result and logged here so
    cache accounting stays observable without being surfaced in ``TokenUsage``.
    """
    if final_text:
        answer_tokens = min(await count_tokens(final_text), output_tokens)
    else:
        answer_tokens = 0
    reasoning_tokens = max(output_tokens - answer_tokens, 0)

    metrics = cache_metrics or {}
    normalized = NormalizedUsage(
        prompt_tokens=prompt_tokens,
        answer_tokens=answer_tokens,
        reasoning_tokens=reasoning_tokens,
        output_tokens=output_tokens,
        total_tokens=prompt_tokens + output_tokens,
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
