"""Offline regression tests for Gemini empty-response classification.

Gemini was the one provider with no empty-outcome classification: an empty or
reasoning-only response was logged as a warning but returned with error=None,
so the runner scored it as a completed wrong answer and cache resume never
re-attempted it — the same bug class the anthropic client fixed (L5). All four
providers now share ``classify_empty_response``.
"""
import asyncio

from src.models import create_provider
from tests.transport_fakes import _GeminiResponse, _GeminiUsage, drive_gemini


def _drive(response):
    client = create_provider("gemini", model="g", api_key="k")
    try:
        return asyncio.run(drive_gemini(client, response=response))
    finally:
        client.close()


def test_normal_response_has_no_error():
    _, resp = _drive(_GeminiResponse(text="Hi"))
    assert resp.error is None
    assert resp.text == "Hi"


def test_empty_zero_usage_is_retryable_error():
    _, resp = _drive(_GeminiResponse(
        text="", usage=_GeminiUsage(candidates_token_count=0, total_token_count=10)))
    assert resp.error is not None and "empty_response_no_output_tokens" in resp.error


def test_reasoning_only_stop_is_retryable_error():
    # Thinking tokens spent, no answer text, normal STOP: a real dropout.
    _, resp = _drive(_GeminiResponse(
        text="",
        usage=_GeminiUsage(candidates_token_count=0, thoughts_token_count=50,
                           total_token_count=60),
        finish_reason="FinishReason.STOP",
    ))
    assert resp.error is not None and "empty_response_reasoning_only" in resp.error


def test_max_tokens_exhaustion_is_scoreable_not_error():
    # All-reasoning budget exhaustion is a real outcome: score it, don't retry.
    _, resp = _drive(_GeminiResponse(
        text="",
        usage=_GeminiUsage(candidates_token_count=0, thoughts_token_count=100,
                           total_token_count=110),
        finish_reason="FinishReason.MAX_TOKENS",
    ))
    assert resp.error is None
    assert resp.tokens.reasoning_tokens == 100
