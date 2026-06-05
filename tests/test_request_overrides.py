"""No-network unit tests for the request-override mechanism.

These tests intercept the chat-completions request dict (by monkeypatching the
client's ``create`` call) and never touch the network. They cover request
construction for the documented migration recipes, the protected-field guard,
duplicate-path failure, set-then-unset ordering, override persistence with
secret redaction, and a streaming/usage-parsing regression fixture.
"""
import asyncio
import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.config import load_config, save_config
from src.core.schemas import (
    BenchmarkConfig,
    ExperimentResult,
    ExperimentSummary,
    RequestOverrides,
)
from src.models.openai_api import OpenAIClient
from src.utils.parser import build_config, parse_args
from src.utils.request_overrides import apply_request_overrides, redact_config


# --------------------------------------------------------------------------- #
# Fake streaming primitives (no network)
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


class _Usage:
    def __init__(self, prompt_tokens, completion_tokens, reasoning_tokens, total_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens

        class _Details:
            pass

        details = _Details()
        details.reasoning_tokens = reasoning_tokens
        self.completion_tokens_details = details


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


def _default_chunks():
    """A minimal valid stream: one content chunk + one usage chunk."""
    return [
        _Chunk(choices=[_Choice(_Delta(content="ok"), finish_reason="stop")]),
        _Chunk(usage=_Usage(prompt_tokens=5, completion_tokens=7, reasoning_tokens=0, total_tokens=12)),
    ]


def build_request(model, *, temperature=0.0, max_output_tokens=4096, top_p=None,
                  overrides=None, base_url="https://example.com/v1", chunks=None):
    """Drive OpenAIClient._call_api with a captured, faked create() call."""
    client = OpenAIClient(model=model, api_key="k", base_url=base_url)
    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeStream(chunks if chunks is not None else _default_chunks())

    client.client.chat.completions.create = fake_create

    response = asyncio.run(client._call_api(
        prompt="hi",
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        request_overrides=overrides,
    ))
    return captured, response


def _config_overrides(argv):
    """Run the CLI->config pipeline and return the merged request_overrides dict."""
    return build_config(parse_args(argv))["request_overrides"]


def build_and_validate(argv):
    """Full CLI->config->validated BenchmarkConfig pipeline (triggers the guard)."""
    cfg = build_config(parse_args(argv))
    cfg.pop("cache", None)
    return BenchmarkConfig(**cfg)


BASE_ARGV = ["--model", "m", "--api-key", "k", "--base-url", "https://x/v1",
             "--max-output-tokens", "100"]

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


# --------------------------------------------------------------------------- #
# AC-1 / AC-2: request construction for the documented recipes
# --------------------------------------------------------------------------- #
def test_uniform_base_request_no_overrides():
    captured, _ = build_request("o3-mini")
    assert set(captured) == {"model", "messages", "temperature", "max_tokens", "stream", "stream_options"}
    assert captured["max_tokens"] == 4096
    # No automatic reasoning-model rename or sniffing fields.
    assert "max_completion_tokens" not in captured
    assert "extra_body" not in captured


def test_vllm_qwen_thinking_typed_boolean():
    ro = _config_overrides(BASE_ARGV + [
        "--request-set", "extra_body.chat_template_kwargs.enable_thinking=true",
    ])
    captured, _ = build_request("qwen3-4b", overrides=ro)
    value = captured["extra_body"]["chat_template_kwargs"]["enable_thinking"]
    assert value is True  # boolean, not the string "true"


def test_top_p_float_and_unset_temperature():
    ro = _config_overrides(BASE_ARGV + ["--request-set", "top_p=0.9", "--request-unset", "temperature"])
    captured, _ = build_request("m", overrides=ro)
    assert captured["top_p"] == 0.9
    assert "temperature" not in captured


def test_deepseek_thinking_with_temperature_drop():
    ro = _config_overrides(BASE_ARGV + [
        "--request-set", "extra_body.thinking.type=enabled",
        "--request-unset", "temperature",
    ])
    captured, _ = build_request("deepseek-chat", overrides=ro)
    assert captured["extra_body"]["thinking"]["type"] == "enabled"
    assert "temperature" not in captured


def test_openrouter_reasoning():
    ro = _config_overrides(BASE_ARGV + ["--request-set", "extra_body.reasoning.enabled=true"])
    captured, _ = build_request("openai/gpt-4o", overrides=ro)
    assert captured["extra_body"]["reasoning"]["enabled"] is True


def test_openai_reasoning_token_redirect_via_placeholder():
    ro = _config_overrides(BASE_ARGV + [
        "--request-set", "max_completion_tokens=${max_output_tokens}",
        "--request-unset", "max_tokens",
        "--request-unset", "temperature",
    ])
    captured, _ = build_request("o3-mini", max_output_tokens=5000, overrides=ro)
    assert captured["max_completion_tokens"] == 5000  # int budget, not a string
    assert isinstance(captured["max_completion_tokens"], int)
    assert "max_tokens" not in captured
    assert "temperature" not in captured


def test_yaml_and_cli_produce_identical_request():
    cli_ro = _config_overrides(BASE_ARGV + [
        "--request-set", "extra_body.chat_template_kwargs.enable_thinking=true",
        "--request-unset", "temperature",
    ])

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.yaml"
        cfg_path.write_text(
            "model: m\n"
            "api_key: k\n"
            "base_url: https://x/v1\n"
            "max_output_tokens: 100\n"
            "request_overrides:\n"
            "  set:\n"
            "    extra_body.chat_template_kwargs.enable_thinking: true\n"
            "  unset:\n"
            "    - temperature\n",
            encoding="utf-8",
        )
        yaml_ro = build_config(parse_args(["--config", str(cfg_path)]))["request_overrides"]

    assert cli_ro == yaml_ro
    captured_cli, _ = build_request("m", overrides=cli_ro)
    captured_yaml, _ = build_request("m", overrides=yaml_ro)
    assert captured_cli == captured_yaml


# --------------------------------------------------------------------------- #
# AC-1 negative: CLI parsing failures
# --------------------------------------------------------------------------- #
def test_request_set_missing_equals_rejected():
    with pytest.raises(ValueError):
        build_config(parse_args(BASE_ARGV + ["--request-set", "extra_body.foo"]))


def test_request_set_json_non_object_rejected():
    with pytest.raises(ValueError):
        build_config(parse_args(BASE_ARGV + ["--request-set-json", "[1,2]"]))


def test_duplicate_cli_path_fails_fast():
    with pytest.raises(ValueError):
        build_config(parse_args(BASE_ARGV + ["--request-set", "top_p=0.1", "--request-set", "top_p=0.2"]))


def test_empty_path_rejected():
    with pytest.raises(ValidationError):
        build_and_validate(BASE_ARGV + ["--request-set-json", '{"": 1}'])


def test_reasoning_recipe_drops_top_p_when_set():
    # Full OpenAI reasoning recipe also unsets top_p (matches the old behavior
    # of suppressing sampling params for reasoning models).
    ro = _config_overrides(BASE_ARGV + [
        "--request-set", "max_completion_tokens=${max_output_tokens}",
        "--request-unset", "max_tokens",
        "--request-unset", "temperature",
        "--request-unset", "top_p",
    ])
    captured, _ = build_request("o3-mini", max_output_tokens=5000, top_p=0.9, overrides=ro)
    assert captured["max_completion_tokens"] == 5000
    assert "top_p" not in captured
    assert "temperature" not in captured
    assert "max_tokens" not in captured


def test_removed_flags_are_unrecognized():
    with pytest.raises(SystemExit):
        parse_args(BASE_ARGV + ["--enable-thinking", "true"])
    with pytest.raises(SystemExit):
        parse_args(BASE_ARGV + ["--reasoning-effort", "high"])


@pytest.mark.parametrize("removed_key, value", [
    ("reasoning_effort", "medium"),
    ("enable_thinking", "true"),
])
def test_removed_yaml_field_rejected_not_silently_dropped(removed_key, value):
    # Old configs using the removed keys must fail loudly (pydantic would
    # otherwise silently ignore them and run a default request).
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.yaml"
        cfg_path.write_text(
            "provider: chat_completion\n"
            "dataset_path: d.jsonl\n"
            "model: m\n"
            "api_key: k\n"
            "base_url: https://x/v1\n"
            "max_output_tokens: 100\n"
            f"{removed_key}: {value}\n",
            encoding="utf-8",
        )
        with pytest.raises(ValidationError) as exc:
            load_config(str(cfg_path))
    assert removed_key in str(exc.value)


def test_bundled_openai_config_migrated():
    cfg = load_config(str(CONFIGS_DIR / "openai.yaml"), api_key="k")
    ro = cfg.request_overrides
    assert ro.set["reasoning_effort"] == "medium"
    assert ro.set["max_completion_tokens"] == "${max_output_tokens}"
    assert "max_tokens" in ro.unset and "temperature" in ro.unset
    captured, _ = build_request("gpt-5.2", max_output_tokens=128000, overrides=ro)
    assert captured["max_completion_tokens"] == 128000
    assert "max_tokens" not in captured and "temperature" not in captured
    assert captured["reasoning_effort"] == "medium"


def test_bundled_local_config_migrated():
    cfg = load_config(str(CONFIGS_DIR / "local.yaml"))
    ro = cfg.request_overrides
    assert ro.set["extra_body.chat_template_kwargs.enable_thinking"] is True
    captured, _ = build_request("Qwen/Qwen3-4B", overrides=ro)
    assert captured["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_null_removed_field_is_ignored_not_rejected():
    # A null leftover (e.g. from an older serialized config) is not an
    # intentional setting and must not be rejected.
    cfg = BenchmarkConfig(
        dataset_path="d.jsonl", provider="chat_completion", model="m",
        base_url="https://x/v1", api_key="k", max_output_tokens=100,
        reasoning_effort=None, enable_thinking=None,
    )
    assert cfg.model == "m"
    assert cfg.reasoning_effort is None
    assert not hasattr(cfg, "enable_thinking")


def test_historical_result_file_with_removed_fields_loads():
    # Regression: result files written by older versions persist removed
    # config fields (even non-null) and must still load.
    result = _make_result("sk-x", {"set": {"extra_body.foo": 1}, "unset": []})
    data = result.model_dump()
    data["config"]["reasoning_effort"] = "medium"
    data["config"]["enable_thinking"] = True
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "old_result.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        loaded = ExperimentResult.load_from_file(str(path))
    assert loaded.config.model == "m"
    # load_from_file strips legacy keys before validation, so a historical
    # chat_completion result (where reasoning_effort is removed) still loads.
    assert loaded.config.reasoning_effort is None


def test_non_chat_result_reasoning_effort_round_trips():
    # A saved anthropic/openai-responses result must preserve reasoning_effort
    # through save_to_file -> load_from_file (it is still a valid field there).
    cfg = BenchmarkConfig(
        dataset_path="d.jsonl", provider="anthropic", model="claude",
        max_output_tokens=100, reasoning_effort="high",
    )
    summary = ExperimentSummary(
        total_problems=0, correct_count=0, accuracy=0, total_tokens=0,
        total_prompt_tokens=0, total_answer_tokens=0, total_reasoning_tokens=0,
        total_output_tokens=0, avg_tokens_per_problem=0, avg_latency=0, total_duration=0,
    )
    result = ExperimentResult(config=cfg, results=[], summary=summary, dataset_name="d")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "result.json"
        result.save_to_file(str(path))
        loaded = ExperimentResult.load_from_file(str(path))
    assert loaded.config.reasoning_effort == "high"


def test_reasoning_effort_retained_for_non_chat_provider():
    # reasoning_effort remains a working config field for anthropic /
    # openai-responses (applied directly by those clients).
    cfg = BenchmarkConfig(
        dataset_path="d.jsonl", provider="anthropic", model="claude",
        max_output_tokens=100, reasoning_effort="high",
    )
    assert cfg.reasoning_effort == "high"


def test_enable_thinking_warns_and_dropped_for_non_chat_provider(caplog):
    # enable_thinking was a chat_completion-only setting; for other providers it
    # is ignored with a warning (not a hard error).
    with caplog.at_level("WARNING"):
        cfg = BenchmarkConfig(
            dataset_path="d.jsonl", provider="anthropic", model="claude",
            max_output_tokens=100, enable_thinking=True,
        )
    assert not hasattr(cfg, "enable_thinking")
    assert any("enable_thinking" in rec.message for rec in caplog.records)


@pytest.mark.parametrize("field, value", [
    ("reasoning_effort", "high"),
    ("enable_thinking", True),
])
def test_removed_field_still_rejected_for_chat_completion(field, value):
    with pytest.raises(ValidationError):
        BenchmarkConfig(**{
            "dataset_path": "d.jsonl", "provider": "chat_completion", "model": "m",
            "base_url": "https://x/v1", "api_key": "k", "max_output_tokens": 100,
            field: value,
        })


# --------------------------------------------------------------------------- #
# AC-3: protected-field prefix guard
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("argv", [
    ["--request-set", "stream=false"],
    ["--request-set", "model=x"],
    ["--request-unset", "stream_options"],
    ["--request-set", "messages.0.content=hi"],
])
def test_protected_paths_rejected(argv):
    with pytest.raises(ValidationError) as exc:
        build_and_validate(BASE_ARGV + argv)
    # Error names the offending path.
    assert "protected" in str(exc.value).lower()


def test_non_protected_override_passes_guard():
    cfg = build_and_validate(BASE_ARGV + ["--request-set", "extra_body.reasoning.effort=high"])
    assert cfg.request_overrides.set["extra_body.reasoning.effort"] == "high"


def test_guard_enforced_on_yaml_config_path():
    # Regression: the guard must fire for the load_config(config_path=...) /
    # run_benchmark YAML path, not just the CLI build_config path.
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.yaml"
        cfg_path.write_text(
            "provider: chat_completion\n"
            "dataset_path: d.jsonl\n"
            "model: m\n"
            "api_key: k\n"
            "base_url: https://x/v1\n"
            "max_output_tokens: 100\n"
            "request_overrides:\n"
            "  set:\n"
            "    stream: false\n"
            "  unset: []\n",
            encoding="utf-8",
        )
        with pytest.raises(ValidationError) as exc:
            load_config(str(cfg_path))
    assert "protected" in str(exc.value).lower()


# --------------------------------------------------------------------------- #
# AC-1: set-then-unset ordering
# --------------------------------------------------------------------------- #
def test_set_then_unset_ordering():
    overrides = RequestOverrides(set={"extra_body.a": 1, "top_p": 0.5}, unset=["top_p"])
    request = {"model": "m", "messages": [], "max_tokens": 10}
    built = apply_request_overrides(request, overrides, {"max_output_tokens": 10})
    assert built["extra_body"]["a"] == 1
    assert "top_p" not in built  # unset wins over set for the same path


# --------------------------------------------------------------------------- #
# AC-5: persistence with secret redaction
# --------------------------------------------------------------------------- #
def _make_result(api_key, overrides):
    cfg = BenchmarkConfig(
        dataset_path="d.jsonl", provider="chat_completion", model="m",
        base_url="https://x/v1", api_key=api_key, max_output_tokens=100,
        request_overrides=overrides,
    )
    summary = ExperimentSummary(
        total_problems=0, correct_count=0, accuracy=0, total_tokens=0,
        total_prompt_tokens=0, total_answer_tokens=0, total_reasoning_tokens=0,
        total_output_tokens=0, avg_tokens_per_problem=0, avg_latency=0, total_duration=0,
    )
    return ExperimentResult(config=cfg, results=[], summary=summary, dataset_name="d")


def test_results_file_records_overrides():
    result = _make_result("sk-x", {"set": {"extra_body.foo": 1}, "unset": []})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.json"
        result.save_to_file(str(path))
        data = json.loads(path.read_text(encoding="utf-8"))
    assert data["config"]["request_overrides"]["set"]["extra_body.foo"] == 1


def test_secret_override_and_api_key_masked():
    result = _make_result("sk-secret", {"set": {"headers.x-api-key": "SECRET", "extra_body.foo": 1}, "unset": []})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.json"
        result.save_to_file(str(path))
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = path.read_text(encoding="utf-8")
    cfg = data["config"]
    assert cfg["api_key"] == "***MASKED***"
    assert cfg["request_overrides"]["set"]["headers.x-api-key"] == "***MASKED***"
    assert cfg["request_overrides"]["set"]["extra_body.foo"] == 1
    # No unmasked secret leaks anywhere in the saved file.
    assert "SECRET" not in raw


def test_save_config_yaml_redaction():
    cfg = BenchmarkConfig(
        dataset_path="d.jsonl", provider="chat_completion", model="m",
        base_url="https://x/v1", api_key="sk-secret", max_output_tokens=100,
        request_overrides={"set": {"headers.authorization": "Bearer T"}, "unset": []},
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cfg.yaml"
        save_config(cfg, str(path))
        text = path.read_text(encoding="utf-8")
    assert "sk-secret" not in text
    assert "Bearer T" not in text
    assert "***MASKED***" in text


def test_cli_set_overrides_yaml_unset():
    # YAML unsets temperature; CLI re-sets it -> CLI must win (no set-then-unset).
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.yaml"
        cfg_path.write_text(
            "model: m\n"
            "api_key: k\n"
            "base_url: https://x/v1\n"
            "max_output_tokens: 100\n"
            "request_overrides:\n"
            "  set: {}\n"
            "  unset:\n"
            "    - temperature\n",
            encoding="utf-8",
        )
        args = parse_args(["--config", str(cfg_path), "--request-set", "temperature=0.7"])
        ro = build_config(args)["request_overrides"]
    assert "temperature" not in ro["unset"]
    captured, _ = build_request("m", overrides=ro)
    assert captured["temperature"] == 0.7


def test_yaml_unset_without_cli_conflict_still_removes():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.yaml"
        cfg_path.write_text(
            "model: m\n"
            "api_key: k\n"
            "base_url: https://x/v1\n"
            "max_output_tokens: 100\n"
            "request_overrides:\n"
            "  set: {}\n"
            "  unset:\n"
            "    - temperature\n",
            encoding="utf-8",
        )
        ro = build_config(parse_args(["--config", str(cfg_path)]))["request_overrides"]
    captured, _ = build_request("m", overrides=ro)
    assert "temperature" not in captured


def test_cli_temperature_flag_overrides_yaml_unset():
    # `--temperature 0.7` must win over the bundled openai.yaml `unset: [temperature]`,
    # while the YAML's max_tokens->max_completion_tokens redirect still applies.
    args = parse_args(["--config", str(CONFIGS_DIR / "openai.yaml"), "--temperature", "0.7"])
    config = build_config(args)
    ro = config["request_overrides"]
    assert "temperature" not in ro["unset"]
    captured, _ = build_request(
        "gpt-5.2", temperature=config["temperature"], max_output_tokens=128000, overrides=ro
    )
    assert captured["temperature"] == 0.7
    assert "max_tokens" not in captured
    assert captured["max_completion_tokens"] == 128000


def test_cli_top_p_flag_overrides_yaml_set():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.yaml"
        cfg_path.write_text(
            "provider: chat_completion\n"
            "dataset_path: d.jsonl\n"
            "model: m\n"
            "api_key: k\n"
            "base_url: https://x/v1\n"
            "max_output_tokens: 100\n"
            "request_overrides:\n"
            "  set:\n"
            "    top_p: 0.2\n"
            "  unset: []\n",
            encoding="utf-8",
        )
        config = build_config(parse_args(["--config", str(cfg_path), "--top-p", "0.95"]))
    ro = config["request_overrides"]
    assert "top_p" not in ro["set"]
    captured, _ = build_request("m", top_p=config["top_p"], overrides=ro)
    assert captured["top_p"] == 0.95


def test_explicit_cli_request_unset_beats_cli_flag():
    # If the user explicitly asks to unset a path via --request-unset, that wins
    # even when they also pass the standard flag for the same field.
    config = build_config(parse_args(
        BASE_ARGV + ["--temperature", "0.7", "--request-unset", "temperature"]
    ))
    assert "temperature" in config["request_overrides"]["unset"]


def test_token_budget_values_not_masked():
    out = redact_config({
        "api_key": "sk-x",
        "request_overrides": {"set": {
            "max_completion_tokens": "${max_output_tokens}",
            "max_tokens": 5000,
            "access_token": "AKIA-SECRET",
            "token": "T",
        }, "unset": []},
    })
    set_map = out["request_overrides"]["set"]
    assert set_map["max_completion_tokens"] == "${max_output_tokens}"  # preserved
    assert set_map["max_tokens"] == 5000  # preserved
    assert set_map["access_token"] == "***MASKED***"  # credential masked
    assert set_map["token"] == "***MASKED***"


def test_redact_config_helper():
    out = redact_config({
        "api_key": "sk-x",
        "request_overrides": {"set": {"token": "abc", "foo": 1}, "unset": []},
    })
    assert out["api_key"] == "***MASKED***"
    assert out["request_overrides"]["set"]["token"] == "***MASKED***"
    assert out["request_overrides"]["set"]["foo"] == 1


def test_nested_secret_in_override_value_masked():
    # A secret hidden inside an object override value (--request-set-json/YAML)
    # must be masked even though the dotted leaf key is not secret-like.
    result = _make_result("sk-x", {
        "set": {"extra_headers": {"Authorization": "Bearer TOPSECRET", "X-Trace": "ok"}},
        "unset": [],
    })
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.json"
        result.save_to_file(str(path))
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    headers = data["config"]["request_overrides"]["set"]["extra_headers"]
    assert headers["Authorization"] == "***MASKED***"
    assert headers["X-Trace"] == "ok"  # non-secret nested value preserved
    assert "TOPSECRET" not in raw


# --------------------------------------------------------------------------- #
# AC-6: streaming/usage parsing regression fixture (unchanged behavior)
# --------------------------------------------------------------------------- #
def test_streaming_usage_parsing_regression():
    chunks = [
        _Chunk(choices=[_Choice(_Delta(reasoning_content="thinking..."))]),
        _Chunk(choices=[_Choice(_Delta(content="Hello "))]),
        _Chunk(choices=[_Choice(_Delta(content="world"), finish_reason="stop")]),
        _Chunk(usage=_Usage(prompt_tokens=10, completion_tokens=50, reasoning_tokens=20, total_tokens=60)),
    ]
    _, response = build_request("m", overrides=None, chunks=chunks)

    assert response.text == "Hello world"
    assert response.finish_reason == "stop"
    assert response.error is None

    tokens = response.tokens
    assert tokens.prompt_tokens == 10
    assert tokens.output_tokens == 50
    assert tokens.reasoning_tokens == 20
    assert tokens.answer_tokens == 30  # completion - reasoning
    assert tokens.total_tokens == 60


def test_extract_tokens_direct():
    client = OpenAIClient(model="m", api_key="k", base_url="https://x/v1")
    chunk = _Chunk(usage=_Usage(prompt_tokens=3, completion_tokens=11, reasoning_tokens=4, total_tokens=14))
    tokens = client._extract_tokens(chunk)
    assert tokens.prompt_tokens == 3
    assert tokens.output_tokens == 11
    assert tokens.reasoning_tokens == 4
    assert tokens.answer_tokens == 7
    assert tokens.total_tokens == 14
