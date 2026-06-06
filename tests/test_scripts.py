"""Guard: bundled benchmark scripts stay valid against the current CLI.

These paper-era scripts are not the tool's interface, but they must not be broken
by the refactor: no removed flags, and math invocations configure the required
LLM judge.
"""
import socket
from pathlib import Path

import pytest

from src.core.inspect import build_inspection
from src.core.schemas import BenchmarkConfig
from src.utils.parser import build_config, parse_args

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
SCRIPTS = sorted(SCRIPTS_DIR.glob("run_*.sh"))
REMOVED_FLAGS = ("--reasoning-effort", "--enable-thinking")


def _strip_comments(text: str) -> str:
    # Drop everything after the first '#' on each line (URLs use '://', no bare '#').
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def test_scripts_exist():
    names = {p.name for p in SCRIPTS}
    assert {"run_openai_benchmark.sh", "run_claude_benchmark.sh",
            "run_opensource_benchmark.sh"} <= names


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_no_removed_flags(script):
    code = _strip_comments(script.read_text(encoding="utf-8"))
    for flag in REMOVED_FLAGS:
        assert flag not in code, f"{script.name} uses removed flag {flag}"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_math_runs_configure_judge(script):
    # Every bundled script runs math (loops all tasks), so it must configure the
    # required judge (model + endpoint) for the math invocation.
    code = script.read_text(encoding="utf-8")
    assert "--judge-model" in code, script.name
    assert "--judge-base-url" in code, script.name


def test_openai_script_uses_full_reasoning_recipe():
    # The OpenAI reasoning runs must redirect the budget and drop sampling fields,
    # not merely set reasoning_effort.
    code = (SCRIPTS_DIR / "run_openai_benchmark.sh").read_text(encoding="utf-8")
    assert "reasoning_effort=$effort" in code
    assert "max_completion_tokens=${max_output_tokens}" in code
    assert "--request-unset max_tokens" in code
    assert "--request-unset temperature" in code


def test_openai_reasoning_invocation_resolves_valid_request(monkeypatch):
    # Reconstruct what the OpenAI script emits for a non-`none` effort math run and
    # confirm the resolved request is valid for a reasoning model (no socket).
    def _boom(*a, **k):
        raise AssertionError("opened a socket during inspect")
    monkeypatch.setattr(socket, "socket", _boom)

    argv = [
        "--provider", "chat_completion", "--model", "gpt-5.4",
        "--api-key", "sk-x", "--base-url", "https://api.openai.com/v1",
        "--task", "math", "--max-output-tokens", "128000",
        "--request-set", "reasoning_effort=high",
        "--request-set", "max_completion_tokens=${max_output_tokens}",
        "--request-unset", "max_tokens",
        "--request-unset", "temperature",
        "--judge-model", "gpt-4o-mini", "--judge-base-url", "https://api.openai.com/v1",
        "--judge-api-key", "sk-x",
    ]
    cfg = BenchmarkConfig(**build_config(parse_args(argv)))
    req = build_inspection(cfg)["request"]
    assert req["max_completion_tokens"] == 128000
    assert req["reasoning_effort"] == "high"
    assert "max_tokens" not in req
    assert "temperature" not in req
