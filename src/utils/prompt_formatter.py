"""
Prompt formatting utilities for enforcing consistent output formats.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


DEFAULT_FORMAT_INSTRUCTION = """After solving the problem, clearly state your final answer at the end in the format: "The answer is [NUMBER]." """

DEFAULT_CODE_FORMAT_INSTRUCTION = """Please provide your solution as executable Python code in a markdown code block. Look at the test cases to determine the exact function name and parameters required.

Example format:
```python
def function_name(params):
    # Your implementation
    pass
```"""


def format_prompt(
    problem: str,
    enforce_format: bool = False,
    custom_instruction: Optional[str] = None,
    evaluator_type: str = "math"
) -> str:
    """
    Format a problem prompt with optional format enforcement instructions.
    
    Args:
        problem: The problem text
        enforce_format: Whether to add format enforcement instructions
        custom_instruction: Custom format instruction (overrides default)
        evaluator_type: Type of evaluator ('math' or 'code')
    
    Returns:
        str: Formatted prompt
    """
    if not enforce_format:
        return problem
    
    # Use custom instruction if provided
    if custom_instruction:
        instruction = custom_instruction
    else:
        # Select default instruction based on evaluator type
        if evaluator_type == "code":
            instruction = DEFAULT_CODE_FORMAT_INSTRUCTION
        else:
            instruction = DEFAULT_FORMAT_INSTRUCTION
    
    # Simply prepend instruction to problem
    formatted = f"{instruction}\n\n{problem}"
    
    logger.debug(f"Formatted prompt with {evaluator_type} format enforcement instruction")
    return formatted

