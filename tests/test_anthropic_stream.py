"""Offline regression tests for Anthropic stream classification and the
count_tokens capability fallback.

Empty-response classification: a 200 stream that ends without visible text used
to be recorded as a successful (wrong) answer, which the cache then treated as
completed — resume never re-attempted it. These tests pin the classification:
zero-output empties and reasoning-only end_turns are retryable errors, while
max_tokens exhaustion stays a scoreable success.

count_tokens capability: a relay serving /messages but not
/messages/count_tokens must be probed once, not once per problem.
"""
import asyncio

import httpx

from src.models.anthropic_api import AnthropicClient
from tests.transport_fakes import _sse, anthropic_sse, drive_anthropic

CLIENT_KWARGS = dict(model="claude", api_key="k", base_url="https://x")


def _client(**overrides):
    return AnthropicClient(**{**CLIENT_KWARGS, **overrides})


def _empty_sse(output_tokens, stop="end_turn"):
    return _sse([
        {"type": "message_start", "message": {"usage": {"input_tokens": 10}, "model": "claude"}},
        {"type": "message_delta", "usage": {"output_tokens": output_tokens},
         "delta": {"stop_reason": stop}},
    ])


# --------------------------------------------------------------------------- #
# Empty-response classification
# --------------------------------------------------------------------------- #
def test_normal_response_has_no_error():
    _, resp = asyncio.run(drive_anthropic(_client(), body=anthropic_sse()))
    assert resp.error is None
    assert resp.text == "Hi"


def test_empty_end_turn_zero_output_is_retryable_error():
    # The observed relay signature: end_turn, no content blocks, zero usage.
    _, resp = asyncio.run(drive_anthropic(_client(), body=_empty_sse(output_tokens=0)))
    assert resp.error is not None and "empty_response_no_output_tokens" in resp.error
    assert resp.text == ""
    assert resp.finish_reason == "end_turn"


def test_stream_with_no_events_is_retryable_error():
    # Degenerate 200 with an empty event stream must not read as success.
    _, resp = asyncio.run(drive_anthropic(_client(), body=_sse([])))
    assert resp.error is not None and "empty_response_no_output_tokens" in resp.error


def test_empty_max_tokens_zero_output_is_still_error():
    # Zero output tokens is the broken-relay signature regardless of stop_reason.
    _, resp = asyncio.run(drive_anthropic(
        _client(), body=_empty_sse(output_tokens=0, stop="max_tokens")))
    assert resp.error is not None and "empty_response_no_output_tokens" in resp.error


def test_max_tokens_exhaustion_is_scoreable_not_error():
    # All-reasoning budget exhaustion is a real outcome: score it, don't retry it.
    _, resp = asyncio.run(drive_anthropic(
        _client(), body=_empty_sse(output_tokens=128000, stop="max_tokens")))
    assert resp.error is None
    assert resp.finish_reason == "max_tokens"
    assert resp.tokens.reasoning_tokens == 128000


def test_reasoning_only_end_turn_is_retryable_error():
    _, resp = asyncio.run(drive_anthropic(
        _client(), body=_empty_sse(output_tokens=50, stop="end_turn")))
    assert resp.error is not None and "empty_response_reasoning_only" in resp.error


# --------------------------------------------------------------------------- #
# count_tokens capability memoization
# --------------------------------------------------------------------------- #
class _FakePostClient:
    """Replays canned count_tokens responses; specs are a status int or a JSON dict."""

    def __init__(self, specs):
        self._specs = specs
        self.post_calls = 0

    async def post(self, url, headers=None, json=None, timeout=None):
        spec = self._specs[min(self.post_calls, len(self._specs) - 1)]
        self.post_calls += 1
        request = httpx.Request("POST", url)
        if isinstance(spec, int):
            return httpx.Response(spec, request=request)
        return httpx.Response(200, request=request, json=spec)


def _count(client, text="hello world"):
    return asyncio.run(client._count_tokens_exact(text))


def test_count_tokens_exact_on_success():
    client = _client()
    client._http_client = _FakePostClient([{"input_tokens": 7}])
    assert _count(client) == (7, False)


def test_count_tokens_404_memoizes_unsupported():
    client = _client()
    fake = _FakePostClient([404])
    client._http_client = fake

    count, estimated = _count(client)
    assert estimated is True and count > 0
    assert client._count_tokens_supported is False

    # Subsequent calls must not re-probe the dead route.
    _count(client)
    _count(client)
    assert fake.post_calls == 1


def test_count_tokens_transient_failure_keeps_probing():
    # A 500 falls back for this call but must not disable the capability.
    client = _client()
    fake = _FakePostClient([500, {"input_tokens": 9}])
    client._http_client = fake

    count, estimated = _count(client)
    assert estimated is True and count > 0
    assert client._count_tokens_supported is True

    assert _count(client) == (9, False)
    assert fake.post_calls == 2


def test_empty_text_skips_probe_entirely():
    client = _client()
    fake = _FakePostClient([{"input_tokens": 5}])
    client._http_client = fake
    assert _count(client, text="") == (0, False)
    assert fake.post_calls == 0
