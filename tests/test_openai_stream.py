"""Offline regression tests for Chat Completions finish_reason fidelity.

``finish_reason`` is provider-literal: it is only "stop" when the provider sent
a stop finish chunk. A stream that ends without any finish chunk (an abnormal
EOF or a reasoning-only stream from a relay) stays ``None`` — fabricating
"stop" made the cached row contradict its own error text and polluted
finish-reason audits.
"""
import asyncio

from src.models.openai_api import OpenAIClient
from tests.transport_fakes import _Choice, _Chunk, _Delta, _Usage, drive_chat, openai_chunks

CLIENT_KWARGS = dict(model="m", api_key="k", base_url="https://x/v1")


def _drive(chunks):
    client = OpenAIClient(**CLIENT_KWARGS)
    return asyncio.run(drive_chat(client, chunks=chunks))


def test_real_stop_chunk_reports_stop():
    _, resp = _drive(openai_chunks(text="ok", finish="stop"))
    assert resp.finish_reason == "stop"
    assert resp.error is None


def test_reasoning_only_stream_without_finish_chunk_is_unknown_not_stop():
    # The observed relay signature: reasoning deltas and usage, then EOF with
    # no finish chunk. The error says unknown; the field must agree (None),
    # not claim a normal stop.
    chunks = [
        _Chunk(choices=[_Choice(_Delta(reasoning_content="thinking..."))]),
        _Chunk(usage=_Usage(prompt_tokens=5, completion_tokens=7,
                            reasoning_tokens=7, total_tokens=12)),
    ]
    _, resp = _drive(chunks)
    assert resp.error is not None and "empty_response_reasoning_only" in resp.error
    assert "finish_reason=unknown" in resp.error
    assert resp.finish_reason is None


def test_text_without_finish_chunk_keeps_none():
    # Success text but no terminal chunk: keep the text, don't fabricate stop.
    chunks = [
        _Chunk(choices=[_Choice(_Delta(content="42"))]),
        _Chunk(usage=_Usage(prompt_tokens=5, completion_tokens=3,
                            reasoning_tokens=0, total_tokens=8)),
    ]
    _, resp = _drive(chunks)
    assert resp.text == "42"
    assert resp.error is None
    assert resp.finish_reason is None
