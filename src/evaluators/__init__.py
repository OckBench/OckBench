"""Evaluators for different task types."""
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class EvalResult:
    """Unified evaluation result from all evaluators."""
    is_correct: bool
    extracted_answer: Optional[Any]
    extraction_method: str
    tests_passed: Optional[int] = None
    tests_total: Optional[int] = None
    execution_error: Optional[str] = None
