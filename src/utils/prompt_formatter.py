"""Prompt formatting utilities for wrapping problems with task-specific instructions."""
import hashlib
from typing import Dict

# Bump when format_prompt's behavior changes in a way not captured by the raw
# template text (e.g. the code test-case injection logic). Part of run identity.
PROMPT_FORMATTER_VERSION = "1"

MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem. Enclose your final answer strictly inside "
    "<answer> and </answer> tags at the very end of your response.\n\n"
    "Problem: {problem}"
)

CODE_PROMPT_TEMPLATE = (
    "Write a Python function to solve the following problem. "
    "Enclose your final executable Python code strictly inside "
    "<solution> and </solution> tags.\n\n"
    "Problem:\n{problem}"
)

SCIENCE_PROMPT_TEMPLATE = (
    "Answer the following multiple-choice question. "
    "Enclose your final chosen letter (A, B, C, or D) strictly inside "
    "<answer> and </answer> tags.\n\n"
    "Question:\n{problem}"
)


_TEMPLATES = {
    "code": CODE_PROMPT_TEMPLATE,
    "science": SCIENCE_PROMPT_TEMPLATE,
    "math": MATH_PROMPT_TEMPLATE,
}


def template_identity(evaluator_type: str) -> Dict[str, str]:
    """Deterministic identity of the prompt/template for a task type.

    Used in the cache run-identity so a template (or formatter-behavior) change
    cannot silently resume into an old cache. A built-in task hashes its template;
    an unknown/custom evaluator uses a neutral ``"raw"`` template (matching
    ``format_prompt``), and the evaluator name keeps the identity distinct.
    """
    template = _TEMPLATES.get(evaluator_type)
    template_hash = (
        hashlib.sha256(template.encode("utf-8")).hexdigest() if template is not None else "raw"
    )
    return {
        "evaluator_type": evaluator_type,
        "formatter_version": PROMPT_FORMATTER_VERSION,
        "template_hash": template_hash,
    }


def format_prompt(problem: str, evaluator_type: str = "math", test_cases=None) -> str:
    """Wrap a problem with task-specific format instructions."""
    if evaluator_type == "code":
        problem_text = problem
        if test_cases:
            if "Your code must be able to pass these tests:" not in problem and "assert " not in problem:
                test_cases_str = "\n".join(test_cases)
                problem_text = f"{problem}\n\nYour code must be able to pass these tests:\n{test_cases_str}"
        return CODE_PROMPT_TEMPLATE.format(problem=problem_text)

    template = _TEMPLATES.get(evaluator_type)
    if template is None:
        # Unknown/custom evaluator (registered via the extension point): use a
        # neutral raw fallback rather than imposing the built-in math prompt, so a
        # custom non-math task is not given inappropriate instructions.
        return problem
    return template.format(problem=problem)
