"""Offline transport fakes for driving each provider client without a network.

Every helper monkeypatches the client's transport, drives ``generate`` (which
shapes the request and dispatches it), and returns ``(captured_request,
response)`` so tests can assert both the outgoing request shape and the parsed
response — all without a socket or an API key.
"""
import json
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from src.core.schemas import ModelResponse


class EmptyStrError(Exception):
    """A retryable exception whose str() is '' (seen from SDK timeouts)."""

    def __str__(self):
        return ""


# --------------------------------------------------------------------------- #
# chat_completion (AsyncOpenAI streaming) fakes
# --------------------------------------------------------------------------- #
class _Delta:
    def __init__(self, content=None, reasoning_content=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.model_extra = None


class _Choice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _Details:
    def __init__(self, reasoning_tokens):
        self.reasoning_tokens = reasoning_tokens


class _Usage:
    def __init__(self, prompt_tokens, completion_tokens, reasoning_tokens, total_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.completion_tokens_details = _Details(reasoning_tokens)


class _Chunk:
    def __init__(self, choices=None, usage=None, model="fake-model"):
        self.choices = choices or []
        self.usage = usage
        self.model = model


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for chunk in self._chunks:
                yield chunk
        return gen()


def openai_chunks(text="ok", prompt_tokens=5, completion_tokens=7,
                  reasoning_tokens=0, total_tokens=12, finish="stop"):
    return [
        _Chunk(choices=[_Choice(_Delta(content=text), finish_reason=finish)]),
        _Chunk(usage=_Usage(prompt_tokens, completion_tokens, reasoning_tokens, total_tokens)),
    ]


async def drive_chat(client, prompt="hi", max_output_tokens=100, chunks=None) -> Tuple[Dict[str, Any], ModelResponse]:
    captured: Dict[str, Any] = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeStream(chunks if chunks is not None else openai_chunks())

    client.client.chat.completions.create = fake_create
    response = await client.generate(prompt, max_output_tokens)
    return captured, response


# --------------------------------------------------------------------------- #
# Raw-httpx SSE fakes (responses + anthropic)
# --------------------------------------------------------------------------- #
class _FakeStreamResp:
    def __init__(self, body: str, status: int = 200):
        self._body = body
        self.status_code = status
        self.request = SimpleNamespace(method="POST", url="https://fake")

    async def aiter_text(self):
        yield self._body

    async def aread(self):
        return b""

    @property
    def text(self):
        return self._body


class _FakeStreamCtx:
    def __init__(self, resp: _FakeStreamResp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


class _FakeHTTPClient:
    """Captures the streamed request body and replays a canned SSE body."""

    def __init__(self, body: str, status: int = 200):
        self._body = body
        self._status = status
        self.captured: Dict[str, Any] = {}
        self.calls = 0

    def stream(self, method, url, headers=None, json=None):
        self.calls += 1
        self.captured = {"method": method, "url": url, "headers": headers, "json": json}
        return _FakeStreamCtx(_FakeStreamResp(self._body, self._status))


def _sse(events: List[dict], done: bool = True) -> str:
    lines = [f"data: {json.dumps(e)}" for e in events]
    if done:
        lines.append("data: [DONE]")
    return "\n".join(lines) + "\n"


def responses_usage(input_tokens=10, output_tokens=5, total_tokens=15, reasoning_tokens=0):
    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}
    if reasoning_tokens:
        usage["output_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return usage


def responses_sse(text="Hi", input_tokens=10, output_tokens=5, total_tokens=15,
                  reasoning_tokens=0, model="m", status="completed") -> str:
    return _sse([
        {"type": "response.output_text.delta", "delta": text},
        {"type": "response.completed", "response": {
            "model": model, "status": status,
            "usage": responses_usage(input_tokens, output_tokens, total_tokens, reasoning_tokens),
        }},
    ])


async def drive_responses(
    client, prompt="hi", max_output_tokens=100, body=None,
) -> Tuple[Dict[str, Any], ModelResponse]:
    fake = _FakeHTTPClient(body if body is not None else responses_sse())
    client._http_client = fake
    response = await client.generate(prompt, max_output_tokens)
    return fake.captured.get("json", {}), response


def anthropic_sse(text="Hi", input_tokens=10, output_tokens=7, model="claude", stop="end_turn") -> str:
    return _sse([
        {"type": "message_start", "message": {"usage": {"input_tokens": input_tokens}, "model": model}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}},
        {"type": "message_delta", "usage": {"output_tokens": output_tokens}, "delta": {"stop_reason": stop}},
    ])


async def drive_anthropic(client, prompt="hi", max_output_tokens=100, body=None,
                          answer_token_count=3) -> Tuple[Dict[str, Any], ModelResponse]:
    fake = _FakeHTTPClient(body if body is not None else anthropic_sse())
    client._http_client = fake

    async def fake_count(_text):
        return answer_token_count, False
    client._count_tokens_exact = fake_count

    response = await client.generate(prompt, max_output_tokens)
    return fake.captured.get("json", {}), response


# --------------------------------------------------------------------------- #
# Gemini SDK fakes
# --------------------------------------------------------------------------- #
class _GeminiUsage:
    def __init__(self, prompt_token_count=10, candidates_token_count=5,
                 total_token_count=15, thoughts_token_count=0):
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.total_token_count = total_token_count
        self.thoughts_token_count = thoughts_token_count


class _GeminiResponse:
    def __init__(self, text="Hi", usage=None, finish_reason=None):
        self.text = text
        self.usage_metadata = usage or _GeminiUsage()
        self.candidates = (
            [SimpleNamespace(finish_reason=finish_reason, content=None)]
            if finish_reason is not None else []
        )


async def drive_gemini(
    client, prompt="hi", max_output_tokens=100, response=None,
) -> Tuple[Dict[str, Any], ModelResponse]:
    captured: Dict[str, Any] = {}

    async def fake_generate_content(model=None, contents=None, config=None):
        captured.update({"model": model, "contents": contents, "config": config})
        return response if response is not None else _GeminiResponse()

    client.client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=fake_generate_content))
    )
    model_response = await client.generate(prompt, max_output_tokens)
    return captured, model_response
