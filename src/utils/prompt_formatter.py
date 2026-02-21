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
    "Please solve the following problem. You may explain your reasoning first. "
    "However, after solving the problem, you MUST enclose the final executable Python code "
    "within <solution> and </solution> tags. Do not put markdown backticks inside the tags, "
    "just raw Python code.\n\n"
    "{problem}"
)

SCIENCE_PROMPT_TEMPLATE = (
    "After analyzing the question, clearly state your final answer as a single letter "
    "(A, B, C, or D) in the format: \"The answer is [LETTER].\"\n\n"
    "{problem}"
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
            if "Your code should pass these tests:" not in problem and "assert " not in problem:
                test_cases_str = "\n  ".join(test_cases)
                problem_text = f"{problem}\n\nYour code should pass these tests:\n  {test_cases_str}"
        return CODE_PROMPT_TEMPLATE.format(problem=problem_text)

    if evaluator_type == "science":
        return SCIENCE_PROMPT_TEMPLATE.format(problem=problem)

    # Default: math
    return MATH_PROMPT_TEMPLATE.format(problem=problem)
