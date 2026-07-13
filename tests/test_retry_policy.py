"""AC-3: a single, consistent retry & error-handling policy (offline)."""
import asyncio

import pytest

import src.models.base as base_mod
from src.models import create_provider
from src.models.gemini_api import _retry_disabled_http_options
from tests.transport_fakes import EmptyStrError

BUILTIN_CLIENTS = {
    "chat_completion": dict(model="m", api_key="k", base_url="https://x/v1"),
    "openai-responses": dict(model="m", api_key="k", base_url="https://x/v1"),
    "anthropic": dict(model="claude", api_key="k", base_url="https://x"),
    "gemini": dict(model="g", api_key="k"),
}


class _Transient(Exception):
    """A retryable transport error (no status_code, not a non-retryable keyword)."""


class _NonRetryable(Exception):
    def __init__(self, msg):
        super().__init__(msg)
        self.status_code = 401


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def _instant(_seconds):
        return None
    monkeypatch.setattr(base_mod.asyncio, "sleep", _instant)


@pytest.mark.parametrize("provider, kwargs", BUILTIN_CLIENTS.items())
def test_transient_error_retries_exactly_max_retries(provider, kwargs):
    client = create_provider(provider, max_retries=3, **kwargs)
    calls = {"n": 0}

    async def failing(_request):
        calls["n"] += 1
        raise _Transient("temporary upstream hiccup")
    client._dispatch = failing

    resp = asyncio.run(client.generate("hi", 100))
    # Exactly one retry sequence owned by the base layer: no doubled attempts.
    assert calls["n"] == 3
    assert resp.error is not None and resp.text == ""


@pytest.mark.parametrize("provider, kwargs", BUILTIN_CLIENTS.items())
def test_non_retryable_error_surfaces_immediately(provider, kwargs):
    client = create_provider(provider, max_retries=3, **kwargs)
    calls = {"n": 0}

    async def auth_fail(_request):
        calls["n"] += 1
        raise _NonRetryable("authentication failed")
    client._dispatch = auth_fail

    resp = asyncio.run(client.generate("hi", 100))
    assert calls["n"] == 1  # not retried
    assert resp.error is not None


@pytest.mark.parametrize("provider, kwargs", BUILTIN_CLIENTS.items())
def test_empty_str_exception_exhaustion_yields_nonempty_error(provider, kwargs):
    # An exhausted retryable exception with str(e) == '' must not surface an
    # empty error string: '' is falsy, so the runner would score the empty
    # text and the cache would mark the problem completed (never retried).
    client = create_provider(provider, max_retries=2, **kwargs)

    async def failing(_request):
        raise EmptyStrError()
    client._dispatch = failing

    resp = asyncio.run(client.generate("hi", 100))
    assert resp.error == "EmptyStrError"
    assert resp.finish_reason == "error"


def test_wall_clock_timeout_retries_exactly_max_retries():
    client = create_provider(
        "chat_completion",
        model="m",
        api_key="k",
        base_url="https://x/v1",
        max_retries=3,
        wall_clock_timeout=0.001,
    )
    calls = {"n": 0}

    async def hangs(_request):
        calls["n"] += 1
        await asyncio.Future()

    client._dispatch = hangs

    resp = asyncio.run(client.generate("hi", 100))
    assert calls["n"] == 3
    assert resp.error == "wall_clock_timeout: attempt exceeded 0.001s"
    assert resp.text == ""


def test_asyncio_timeout_without_wall_clock_keeps_generic_error():
    client = create_provider(
        "chat_completion",
        model="m",
        api_key="k",
        base_url="https://x/v1",
        max_retries=1,
    )

    async def timeout(_request):
        raise asyncio.TimeoutError()

    client._dispatch = timeout

    resp = asyncio.run(client.generate("hi", 100))
    assert resp.error == "TimeoutError"


def test_chat_completion_disables_sdk_retry():
    client = create_provider("chat_completion", model="m", api_key="k", base_url="https://x/v1")
    assert client.client.max_retries == 0


def test_gemini_disables_sdk_retry():
    opts = _retry_disabled_http_options()
    # Typed HttpOptions or the dict fallback — both must encode a single attempt.
    attempts = None
    retry_options = getattr(opts, "retry_options", None)
    if retry_options is not None:
        attempts = getattr(retry_options, "attempts", None)
    elif isinstance(opts, dict):
        attempts = opts.get("retry_options", {}).get("attempts")
    assert attempts == 1


def test_raw_http_400_surfaces_immediately_for_responses_and_anthropic():
    # A real HTTP 400 from the raw-httpx providers must carry status_code so the
    # base layer treats it as non-retryable (no stacking, no retry).
    from tests.transport_fakes import _FakeHTTPClient

    for provider, kwargs, url in [
        ("openai-responses", BUILTIN_CLIENTS["openai-responses"], None),
        ("anthropic", BUILTIN_CLIENTS["anthropic"], None),
    ]:
        client = create_provider(provider, max_retries=3, **kwargs)
        fake = _FakeHTTPClient(body="bad request", status=400)
        client._http_client = fake
        if provider == "anthropic":
            async def _count(_t):
                return 1, False
            client._count_tokens_exact = _count
        import asyncio as _a
        resp = _a.run(client.generate("hi", 100))
        assert fake.calls == 1, provider          # not retried
        assert resp.error is not None and "400" in resp.error


def test_gemini_uses_native_async_sdk():
    # Regression: google-genai exposes a native async client; do not wrap the
    # synchronous API in an executor and cap concurrency on a thread pool.
    import inspect as _i

    from src.models.gemini_api import GeminiClient
    client = create_provider("gemini", model="g", api_key="k")
    try:
        src = _i.getsource(GeminiClient._generate_content_async)
        assert "client.aio.models.generate_content" in src
        assert "run_in_executor" not in src
    finally:
        client.close()


def test_base_is_single_retry_owner():
    # No provider client defines its own generate(); they all inherit the base.
    import src.models.base as bm
    from src.models.anthropic_api import AnthropicClient
    from src.models.gemini_api import GeminiClient
    from src.models.openai_api import OpenAIClient
    from src.models.openai_responses_api import OpenAIResponsesClient
    for cls in (OpenAIClient, OpenAIResponsesClient, AnthropicClient, GeminiClient):
        assert "generate" not in cls.__dict__
        assert cls.generate is bm.BaseModelClient.generate
