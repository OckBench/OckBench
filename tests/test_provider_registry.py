"""AC-1: uniform, registry-based provider abstraction (offline)."""
import asyncio
import inspect as _inspect

import pytest

import src.models.registry as registry
from src.core.schemas import ModelResponse, TokenUsage
from src.models import available_providers, create_provider
from src.models.base import BaseModelClient
from src.models.openai_api import OpenAIClient
from tests.transport_fakes import drive_anthropic, drive_chat, drive_gemini, drive_responses

BUILTINS = ["chat_completion", "openai-responses", "anthropic", "gemini"]


def test_four_builtins_registered():
    for name in BUILTINS:
        assert name in available_providers()


def test_each_builtin_resolves_and_responds_via_registry():
    # chat_completion
    c = create_provider("chat_completion", model="m", api_key="k", base_url="https://x/v1")
    _, resp = asyncio.run(drive_chat(c))
    assert isinstance(resp, ModelResponse) and resp.text == "ok"

    # openai-responses
    c = create_provider("openai-responses", model="m", api_key="k", base_url="https://x/v1")
    _, resp = asyncio.run(drive_responses(c))
    assert resp.text == "Hi" and resp.error is None

    # anthropic
    c = create_provider("anthropic", model="claude", api_key="k", base_url="https://x")
    _, resp = asyncio.run(drive_anthropic(c))
    assert resp.text == "Hi" and resp.error is None

    # gemini
    c = create_provider("gemini", model="g", api_key="k")
    _, resp = asyncio.run(drive_gemini(c))
    assert resp.text == "Hi"


def test_unknown_provider_enumerates_registered():
    with pytest.raises(ValueError) as exc:
        create_provider("does-not-exist", model="m")
    msg = str(exc.value)
    assert "does-not-exist" in msg
    for name in BUILTINS:
        assert name in msg  # error enumerates available providers


def test_external_provider_registered_via_extension_point():
    name = "test-echo-provider"
    try:
        @registry.register_provider(name)
        class _EchoClient(BaseModelClient):
            protected_paths = ("model",)
            provider_name = name

            def build_request(self, prompt, max_output_tokens):
                return {"model": self.model, "prompt": prompt, "budget": max_output_tokens}

            async def _dispatch(self, request):
                return ModelResponse(
                    text=f"echo:{request['prompt']}",
                    tokens=TokenUsage(prompt_tokens=1, answer_tokens=1, reasoning_tokens=0,
                                      output_tokens=1, total_tokens=2),
                    latency=0, model=self.model,
                )

        assert name in available_providers()
        client = create_provider(name, model="m")  # selectable by name, no core edits
        _, resp = client, asyncio.run(client.generate("hello", 10))
        assert resp.text == "echo:hello"
    finally:
        registry._PROVIDER_REGISTRY.pop(name, None)


def test_runner_constructs_client_through_registry():
    # The runner must build its client via create_provider (no per-provider
    # if/elif). Assert _create_client delegates to the registry.
    import src.core.runner as runner_mod

    src = _inspect.getsource(runner_mod.BenchmarkRunner._create_client)
    assert "create_provider" in src
    # No resurrected dispatch chain.
    assert "elif self.config.provider" not in src


def test_duplicate_registration_rejected():
    with pytest.raises(ValueError):
        registry.register_provider("chat_completion")(OpenAIClient)
