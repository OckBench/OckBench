"""CLI -> config assembly behaviors (offline)."""
import tempfile
from pathlib import Path

import pytest

from src.core.schemas import BenchmarkConfig
from src.evaluators import get_evaluator
from src.utils.parser import build_config, parse_args

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
_KEY_ENV_VARS = ("JUDGE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY")


@pytest.fixture(autouse=True)
def _clean_key_env(monkeypatch):
    """Make env-key resolution deterministic regardless of the ambient shell."""
    for var in _KEY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _yaml(tmp, text):
    p = Path(tmp) / "c.yaml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_config_provider_not_clobbered_by_cli_default():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _yaml(tmp,
            "provider: anthropic\ndataset_path: d.jsonl\nmodel: claude\n"
            "base_url: https://x\nmax_output_tokens: 100\nevaluator_type: math\n")
        config = build_config(parse_args(["--config", cfg_path]))
    assert config["provider"] == "anthropic"  # not overridden to chat_completion


def test_provider_defaults_to_chat_completion_when_unset():
    config = build_config(parse_args(["--model", "m", "--api-key", "k",
                                      "--base-url", "https://x/v1", "--max-output-tokens", "100"]))
    assert config["provider"] == "chat_completion"


def test_cli_provider_overrides_config():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _yaml(tmp,
            "provider: anthropic\ndataset_path: d.jsonl\nmodel: m\n"
            "base_url: https://x\nmax_output_tokens: 100\n")
        config = build_config(parse_args(["--config", cfg_path, "--provider", "gemini"]))
    assert config["provider"] == "gemini"


def test_judge_flags_assemble_judge_config():
    config = build_config(parse_args([
        "--model", "m", "--api-key", "k", "--base-url", "https://x/v1",
        "--max-output-tokens", "100", "--evaluator-type", "math",
        "--judge-model", "gpt-4o-mini", "--judge-base-url", "https://j/v1",
        "--judge-api-key", "jk",
    ]))
    assert config["judge"] == {"model": "gpt-4o-mini", "base_url": "https://j/v1", "api_key": "jk"}


def test_dataset_split_recorded():
    config = build_config(parse_args([
        "--model", "m", "--api-key", "k", "--base-url", "https://x/v1",
        "--max-output-tokens", "100", "--dataset-split", "Selected",
    ]))
    assert config["dataset_split"] == "Selected"


def test_wall_clock_timeout_cli_flag():
    config = build_config(parse_args([
        "--model", "m", "--api-key", "k", "--base-url", "https://x/v1",
        "--max-output-tokens", "100", "--wall-clock-timeout", "300",
    ]))
    assert config["wall_clock_timeout"] == 300


# --------------------------------------------------------------------------- #
# Env-key resolution on the CLI path (AC-4 judge key, AC-10 provider keys)
# --------------------------------------------------------------------------- #
def test_judge_api_key_resolves_from_judge_env(monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "judge-env")
    config = build_config(parse_args(["--config", str(CONFIGS_DIR / "openai.yaml"), "--api-key", "model-key"]))
    assert config["judge"]["api_key"] == "judge-env"
    assert config["api_key"] == "model-key"  # explicit model key untouched


def test_judge_api_key_falls_back_to_openai_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-env")  # JUDGE_API_KEY absent
    config = build_config(parse_args(["--config", str(CONFIGS_DIR / "openai.yaml"), "--api-key", "model-key"]))
    assert config["judge"]["api_key"] == "openai-env"


def test_judge_api_key_cli_flag_overrides_env(monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "judge-env")
    config = build_config(parse_args([
        "--config", str(CONFIGS_DIR / "openai.yaml"), "--api-key", "model-key",
        "--judge-api-key", "cli-judge",
    ]))
    assert config["judge"]["api_key"] == "cli-judge"


@pytest.mark.parametrize("config_file, env_var", [
    ("openai_responses.yaml", "OPENAI_API_KEY"),
    ("anthropic.yaml", "ANTHROPIC_API_KEY"),
    ("gemini.yaml", "GEMINI_API_KEY"),
])
def test_provider_api_key_resolves_from_env(monkeypatch, config_file, env_var):
    monkeypatch.setenv(env_var, "provider-env")
    config = build_config(parse_args(["--config", str(CONFIGS_DIR / config_file)]))
    assert config["api_key"] == "provider-env"


def test_explicit_api_key_wins_over_provider_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "provider-env")
    config = build_config(parse_args([
        "--config", str(CONFIGS_DIR / "openai_responses.yaml"), "--api-key", "explicit",
    ]))
    assert config["api_key"] == "explicit"


def test_missing_judge_key_still_fails_fast(monkeypatch):
    # No judge/provider env set (autouse cleared them): math must fail fast.
    config = build_config(parse_args(["--config", str(CONFIGS_DIR / "openai.yaml"), "--api-key", "model-key"]))
    assert config["judge"].get("api_key") in (None, "")
    with pytest.raises(ValueError) as exc:
        get_evaluator("math", BenchmarkConfig(**config))
    assert "judge api-key" in str(exc.value)
