"""
Prompt formatting utilities for enforcing consistent output formats.
"""
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


DEFAULT_FORMAT_INSTRUCTION = """After solving the problem, clearly state your final answer at the end in the format: "The answer is [NUMBER]." """

DEFAULT_CODE_FORMAT_INSTRUCTION = """Please solve the following problem. You may explain your reasoning first. However, after solving the problem, you MUST enclose the final executable Python code within <solution> and </solution> tags. Do not put markdown backticks inside the tags, just raw Python code."""

DEFAULT_SCIENCE_FORMAT_INSTRUCTION = """After analyzing the question, clearly state your final answer as a single letter (A, B, C, or D) in the format: "The answer is [LETTER]." """


def format_prompt(
    problem: str,
    enforce_format: bool = False,
    custom_instruction: Optional[str] = None,
    evaluator_type: str = "math",
    test_cases: Optional[List[str]] = None
) -> str:
    """
    Format a problem prompt with optional format enforcement instructions.
    
    Args:
        problem: The problem text
        enforce_format: Whether to add format enforcement instructions
        custom_instruction: Custom format instruction (overrides default)
        evaluator_type: Type of evaluator ('math' or 'code')
        test_cases: List of test case strings (for code evaluation)
    
    Returns:
        str: Formatted prompt
    """
    # Build the problem text with test cases if provided
    problem_text = problem
    
    # For code evaluation, append test cases if they're provided and not already in the problem
    if evaluator_type == "code" and test_cases:
        # Check if test cases are already included in the problem text
        # (e.g., MBPP format already has them)
        if "Your code should pass these tests:" not in problem and "assert " not in problem:
            test_cases_str = "\n  ".join(test_cases)
            problem_text = f"{problem}\n\nYour code should pass these tests:\n  {test_cases_str}"
    
    if not enforce_format:
        return problem_text
    
    # Use custom instruction if provided
    if custom_instruction:
        instruction = custom_instruction
    else:
        # Select default instruction based on evaluator type
        if evaluator_type == "code":
            instruction = DEFAULT_CODE_FORMAT_INSTRUCTION
        elif evaluator_type == "science":
            instruction = DEFAULT_SCIENCE_FORMAT_INSTRUCTION
        else:
            instruction = DEFAULT_FORMAT_INSTRUCTION
    
    # Simply prepend instruction to problem
    formatted = f"{instruction}\n\n{problem_text}"
    
    logger.debug(f"Formatted prompt with {evaluator_type} format enforcement instruction")
    return formatted



