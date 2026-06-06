"""AC-9: cross-provider dry-run / inspect with secret safety (offline)."""
import socket
from pathlib import Path

import pytest

from src.core.config import load_config
from src.core.inspect import build_inspection, format_inspection
from src.core.schemas import BenchmarkConfig, JudgeConfig

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _provider_config(provider, **extra):
    base = dict(dataset_path="d.jsonl", model="m", api_key="k", base_url="https://x/v1",
                max_output_tokens=1000, evaluator_type="science", provider=provider)
    if provider == "anthropic":
        base["base_url"] = "https://x"
    if provider == "gemini":
        base.pop("base_url", None)
    base.update(extra)
    return BenchmarkConfig(**base)


@pytest.mark.parametrize("provider", ["chat_completion", "openai-responses", "anthropic", "gemini"])
def test_inspect_runs_with_no_socket(provider, monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("inspect opened a network socket")
    monkeypatch.setattr(socket, "socket", _boom)

    inspection = build_inspection(_provider_config(provider))
    assert inspection["provider"] == provider
    assert "request" in inspection
    # format must also be socket-free
    assert isinstance(format_inspection(inspection), str)


def test_inspect_chat_completion_fixture_resolves_request():
    cfg = load_config(str(CONFIGS_DIR / "openai.yaml"), api_key="k")
    inspection = build_inspection(cfg)
    req = inspection["request"]
    assert req["max_completion_tokens"] == 128000  # resolved from ${max_output_tokens}
    assert req["reasoning_effort"] == "medium"
    assert "temperature" not in req  # dropped via unset
    assert "max_tokens" not in req


def test_inspect_masks_all_secrets():
    cfg = BenchmarkConfig(
        dataset_path="d.jsonl", provider="chat_completion", model="m",
        base_url="https://user:p4ssw0rd@relay.example.com/v1", api_key="sk-TOPSECRET",
        max_output_tokens=1000, evaluator_type="math",
        request_overrides={"set": {"extra_headers.Authorization": "Bearer BEARERSECRET"}, "unset": []},
        judge=JudgeConfig(model="judge-m", base_url="https://api.openai.com/v1", api_key="judge-SECRET"),
    )
    inspection = build_inspection(cfg)
    blob = format_inspection(inspection)

    # No secret of any kind leaks.
    for secret in ("sk-TOPSECRET", "p4ssw0rd", "BEARERSECRET", "judge-SECRET"):
        assert secret not in blob

    assert inspection["api_key"] == "***MASKED***"
    assert "***MASKED***" in inspection["base_url"]  # credentialed base_url userinfo masked
    assert "relay.example.com" in inspection["base_url"]  # host still visible
    assert inspection["request"]["extra_headers"]["Authorization"] == "***MASKED***"
    assert inspection["judge"]["api_key"] == "***MASKED***"
    assert inspection["judge"]["model"] == "judge-m"


def test_inspect_protected_override_still_rejected():
    # Inspect builds the real client, so an illegal override fails fast here too.
    cfg = _provider_config("chat_completion",
                           request_overrides={"set": {"stream": False}, "unset": []})
    with pytest.raises(ValueError):
        build_inspection(cfg)
