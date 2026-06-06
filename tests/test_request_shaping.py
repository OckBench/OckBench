"""AC-2: unified request-shaping across all providers (offline).

Every provider builds its base request, then the shared request_overrides seam
(set/unset, ${max_output_tokens}, per-provider protected-path guard) shapes it.
No provider hard-codes un-overridable reasoning/thinking placement.
"""
import pytest

from src.core.schemas import RequestOverrides
from src.models import create_provider


def _client(provider, overrides=None, **extra):
    kwargs = dict(model="m", api_key="k", base_url="https://x/v1", temperature=0.0)
    if provider == "anthropic":
        kwargs["base_url"] = "https://x"
    if provider == "gemini":
        kwargs.pop("base_url", None)
    kwargs.update(extra)
    if overrides is not None:
        kwargs["request_overrides"] = overrides
    return create_provider(provider, **kwargs)


# --------------------------------------------------------------------------- #
# Reasoning/thinking via the SAME override mechanism for every provider
# --------------------------------------------------------------------------- #
def test_chat_completion_reasoning_override():
    ro = RequestOverrides(set={"reasoning_effort": "high"})
    req = _client("chat_completion", ro).shape_request("hi", 100)
    assert req["reasoning_effort"] == "high"


def test_responses_reasoning_override_replaces_temperature():
    ro = RequestOverrides(set={"reasoning.effort": "high"}, unset=["temperature"])
    req = _client("openai-responses", ro).shape_request("hi", 100)
    assert req["reasoning"]["effort"] == "high"
    assert "temperature" not in req  # not hard-coded; user redirected it


def test_anthropic_thinking_is_overridable_not_hardcoded():
    # Default thinking is present but overridable -> not un-overridable hard-coding.
    default_req = _client("anthropic").shape_request("hi", 100)
    assert default_req["thinking"]["display"] == "summarized"

    ro = RequestOverrides(set={"thinking.type": "enabled", "output_config.effort": "high"})
    req = _client("anthropic", ro).shape_request("hi", 100)
    assert req["thinking"]["type"] == "enabled"
    assert req["output_config"]["effort"] == "high"


def test_gemini_thinking_budget_override():
    ro = RequestOverrides(set={"config.thinking_config.thinking_budget": 2048})
    req = _client("gemini", ro).shape_request("hi", 100)
    assert req["config"]["thinking_config"]["thinking_budget"] == 2048


# --------------------------------------------------------------------------- #
# ${max_output_tokens} substitution keeps the int type, for every provider
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("provider, path, getter", [
    ("chat_completion", "max_completion_tokens", lambda r: r["max_completion_tokens"]),
    ("openai-responses", "metadata.budget", lambda r: r["metadata"]["budget"]),
    ("anthropic", "metadata.budget", lambda r: r["metadata"]["budget"]),
    ("gemini", "config.budget", lambda r: r["config"]["budget"]),
])
def test_max_output_tokens_placeholder_substitution(provider, path, getter):
    ro = RequestOverrides(set={path: "${max_output_tokens}"})
    req = _client(provider, ro).shape_request("hi", 4242)
    value = getter(req)
    assert value == 4242 and isinstance(value, int)


# --------------------------------------------------------------------------- #
# Per-provider protected-path guard: rejected naming field + provider
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("provider, protected", [
    ("chat_completion", "stream"),
    ("chat_completion", "messages"),
    ("openai-responses", "input"),
    ("anthropic", "messages"),
    ("gemini", "model"),
])
def test_protected_override_rejected(provider, protected):
    ro = RequestOverrides(set={protected: "x"})
    with pytest.raises(ValueError) as exc:
        _client(provider, ro)
    msg = str(exc.value).lower()
    assert protected in msg and provider in msg and "protected" in msg


def test_gemini_tolerates_unset_config():
    # config is overridable (nested reasoning lives under it); unsetting it whole
    # must not crash dispatch.
    import asyncio

    from tests.transport_fakes import drive_gemini
    ro = RequestOverrides(unset=["config"])
    client = _client("gemini", ro)
    req = client.shape_request("hi", 100)
    assert "config" not in req
    captured, resp = asyncio.run(drive_gemini(client))
    assert resp.text == "Hi" and captured["config"] == {}


def test_non_protected_override_passes():
    ro = RequestOverrides(set={"extra_body.reasoning.effort": "high"})
    req = _client("chat_completion", ro).shape_request("hi", 100)
    assert req["extra_body"]["reasoning"]["effort"] == "high"


# --------------------------------------------------------------------------- #
# Generation vs connection parameter split
# --------------------------------------------------------------------------- #
def test_generation_vs_connection_split():
    c = _client("chat_completion", temperature=0.3, top_p=0.9, timeout=42, max_retries=2)
    # Generation params shape the request...
    req = c.shape_request("hi", 100)
    assert req["temperature"] == 0.3 and req["top_p"] == 0.9 and req["max_tokens"] == 100
    # ...connection params do not leak into the request.
    assert "timeout" not in req and "max_retries" not in req
    assert c.timeout == 42 and c.max_retries == 2
