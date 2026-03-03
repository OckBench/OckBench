"""Evaluators for different task types."""
from dataclasses import dataclass
from typing import Any, Optional

from .math_eval import MathEvaluator, get_evaluator
from .science_eval import ScienceEvaluator


@dataclass
class EvalResult:
    """Unified evaluation result from all evaluators."""
    is_correct: bool
    extracted_answer: Optional[Any]
    extraction_method: str
    # Code-specific fields (None for math/science)
    tests_passed: Optional[int] = None
    tests_total: Optional[int] = None
    execution_error: Optional[str] = None


__all__ = [
    'MathEvaluator',
    'ScienceEvaluator',
    'get_evaluator',
    'EvalResult',
]
