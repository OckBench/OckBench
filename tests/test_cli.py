"""CLI -> config assembly behaviors (offline)."""
import tempfile
from pathlib import Path

from src.utils.parser import build_config, parse_args


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
