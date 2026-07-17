"""Offline regression tests for Responses API SSE terminal-event handling.

Terminal events (``response.completed`` / ``response.incomplete`` /
``response.failed``) are the authoritative end-of-stream signal; the ``[DONE]``
sentinel is an optional transport convention some proxies append. A provider
that sends ``response.completed`` and then closes the connection (observed on
a relay) is a success, not a transport error — only a stream that ends with no
terminal event at all is ``responses_stream_incomplete``.

``response.incomplete`` keeps its usage/status/reason: budget exhaustion
(``max_output_tokens`` with real output tokens) is a scoreable outcome, matching
the anthropic ``max_tokens`` semantics.
"""
import asyncio

from src.models.openai_responses_api import OpenAIResponsesClient
from tests.transport_fakes import _sse, drive_responses, responses_sse, responses_usage

CLIENT_KWARGS = dict(model="m", api_key="k", base_url="https://x/v1")


def _drive(body):
    client = OpenAIResponsesClient(**CLIENT_KWARGS)
    return asyncio.run(drive_responses(client, body=body))


# --------------------------------------------------------------------------- #
# Terminal-event authority
# --------------------------------------------------------------------------- #
def test_normal_stream_with_done_sentinel_is_success():
    _, resp = _drive(responses_sse(text="Hi"))
    assert resp.error is None
    assert resp.text == "Hi"
    assert resp.finish_reason == "completed"


def test_completed_then_eof_without_done_is_success():
    # The observed relay signature: a full response.completed (answer + usage),
    # then EOF with no [DONE]. Must not be recorded as transport-incomplete.
    body = _sse([
        {"type": "response.output_text.delta", "delta": "42"},
        {"type": "response.completed",
         "response": {"model": "m", "status": "completed", "usage": responses_usage()}},
    ], done=False)
    _, resp = _drive(body)
    assert resp.error is None
    assert resp.text == "42"
    assert resp.finish_reason == "completed"
    assert resp.tokens.output_tokens == 5


def test_truncated_stream_is_transport_incomplete():
    # A genuinely severed stream: deltas, then EOF with no terminal event.
    body = _sse([{"type": "response.output_text.delta", "delta": "partial"}], done=False)
    _, resp = _drive(body)
    assert resp.error is not None and "responses_stream_incomplete" in resp.error
    assert resp.finish_reason is None


def test_done_without_terminal_event_is_transport_incomplete():
    _, resp = _drive(_sse([]))
    assert resp.error is not None and "responses_stream_incomplete" in resp.error


# --------------------------------------------------------------------------- #
# response.incomplete: usage/status/reason preserved
# --------------------------------------------------------------------------- #
def test_incomplete_max_output_tokens_is_scoreable_not_error():
    # Budget exhaustion with real output tokens (all reasoning) is a real
    # outcome: keep the usage, score the empty answer, don't retry.
    body = _sse([
        {"type": "response.incomplete",
         "response": {"model": "m", "status": "incomplete",
                      "incomplete_details": {"reason": "max_output_tokens"},
                      "usage": responses_usage(output_tokens=64, total_tokens=74,
                                               reasoning_tokens=64)}},
    ], done=False)
    _, resp = _drive(body)
    assert resp.error is None
    assert resp.text == ""
    assert resp.finish_reason == "incomplete:max_output_tokens"
    assert resp.tokens.output_tokens == 64
    assert resp.tokens.reasoning_tokens == 64


def test_incomplete_zero_output_is_retryable_error():
    # Zero output tokens is the broken-relay signature regardless of reason.
    body = _sse([
        {"type": "response.incomplete",
         "response": {"model": "m", "status": "incomplete",
                      "incomplete_details": {"reason": "max_output_tokens"},
                      "usage": responses_usage(output_tokens=0, total_tokens=10)}},
    ])
    _, resp = _drive(body)
    assert resp.error is not None and "empty_response_no_output_tokens" in resp.error


def test_incomplete_other_reason_is_retryable_error():
    body = _sse([
        {"type": "response.incomplete",
         "response": {"model": "m", "status": "incomplete",
                      "incomplete_details": {"reason": "content_filter"},
                      "usage": responses_usage(output_tokens=20)}},
    ])
    _, resp = _drive(body)
    assert resp.error is not None and "content_filter" in resp.error
    assert resp.finish_reason == "incomplete:content_filter"


def test_failed_event_is_error():
    body = _sse([
        {"type": "response.failed",
         "response": {"model": "m", "status": "failed",
                      "error": {"message": "server exploded"}}},
    ], done=False)
    _, resp = _drive(body)
    assert resp.error is not None and "responses_stream_failed" in resp.error
    assert "server exploded" in resp.error


# --------------------------------------------------------------------------- #
# reasoning_text deltas
# --------------------------------------------------------------------------- #
def test_reasoning_text_delta_classifies_reasoning_only():
    # MiniMax quirk: reasoning text streamed but reasoning_tokens reported as 0.
    # The collected reasoning deltas are the classification signal.
    body = _sse([
        {"type": "response.reasoning_text.delta", "delta": "thinking..."},
        {"type": "response.completed",
         "response": {"model": "m", "status": "completed",
                      "usage": responses_usage(output_tokens=13, total_tokens=23)}},
    ])
    _, resp = _drive(body)
    assert resp.error is not None and "empty_response_reasoning_only" in resp.error
    assert resp.text == ""


def test_reasoning_delta_does_not_pollute_answer_text():
    body = _sse([
        {"type": "response.reasoning_text.delta", "delta": "let me think"},
        {"type": "response.output_text.delta", "delta": "42"},
        {"type": "response.completed",
         "response": {"model": "m", "status": "completed", "usage": responses_usage()}},
    ])
    _, resp = _drive(body)
    assert resp.error is None
    assert resp.text == "42"
