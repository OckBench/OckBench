"""
Prompt formatting utilities for wrapping problems with task-specific instructions.
"""
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


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


def format_prompt(
    problem: str,
    evaluator_type: str = "math",
    test_cases: Optional[List[str]] = None
) -> str:
    """
    Wrap a problem with task-specific format instructions.

    Args:
        problem: The problem text
        evaluator_type: Type of evaluator ('math', 'code', or 'science')
        test_cases: List of test case strings (for code evaluation)

    Returns:
        str: Formatted prompt ready to send to the model
    """
    if evaluator_type == "code":
        problem_text = problem
        if test_cases:
            if "Your code must be able to pass these tests:" not in problem and "assert " not in problem:
                test_cases_str = "\n".join(test_cases)
                problem_text = f"{problem}\n\nYour code must be able to pass these tests:\n{test_cases_str}"
        return CODE_PROMPT_TEMPLATE.format(problem=problem_text)

    if evaluator_type == "science":
        return SCIENCE_PROMPT_TEMPLATE.format(problem=problem)

    # Default: math
    return MATH_PROMPT_TEMPLATE.format(problem=problem)
