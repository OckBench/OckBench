"""Guard: bundled benchmark scripts stay valid against the current CLI.

These paper-era scripts are not the tool's interface, but they must not be broken
by the refactor: no removed flags, and math invocations configure the required
LLM judge.
"""
from pathlib import Path

import pytest

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
