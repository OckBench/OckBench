"""Prompt formatting: built-in templates + neutral fallback for custom evaluators."""
from src.utils.prompt_formatter import (
    CODE_PROMPT_TEMPLATE,
    MATH_PROMPT_TEMPLATE,
    SCIENCE_PROMPT_TEMPLATE,
    format_prompt,
    template_identity,
)

PROBLEM = "What is 2 + 2?"


def test_builtin_prompts_unchanged():
    assert format_prompt(PROBLEM, "math") == MATH_PROMPT_TEMPLATE.format(problem=PROBLEM)
    assert format_prompt(PROBLEM, "science") == SCIENCE_PROMPT_TEMPLATE.format(problem=PROBLEM)
    # code wraps with the solution template (no test cases here)
    assert format_prompt(PROBLEM, "code") == CODE_PROMPT_TEMPLATE.format(problem=PROBLEM)


def test_custom_evaluator_gets_neutral_raw_prompt():
    out = format_prompt(PROBLEM, "my-custom-task")
    assert out == PROBLEM                                   # raw, no instructions
    assert "Solve the following math problem" not in out    # not the math template
    assert "multiple-choice" not in out


def test_default_is_math():
    # Default evaluator_type stays math (backward-compatible default).
    assert format_prompt(PROBLEM) == MATH_PROMPT_TEMPLATE.format(problem=PROBLEM)


def test_template_identity_raw_for_custom_distinct_from_math():
    custom = template_identity("my-custom-task")
    assert custom["template_hash"] == "raw"
    assert custom["evaluator_type"] == "my-custom-task"
    # Distinct from a built-in template identity, so the cache identity is accurate.
    assert custom["template_hash"] != template_identity("math")["template_hash"]


def test_template_identity_distinct_per_custom_name():
    # The evaluator name keeps custom identities distinct even though both are "raw".
    a = template_identity("task-a")
    b = template_identity("task-b")
    assert a["template_hash"] == b["template_hash"] == "raw"
    assert a != b  # evaluator_type differs
