"""Evaluators for different task types.

Importing this package registers the built-in evaluators (math, science, code)
with the evaluator registry. External tasks register the same way by importing
their own module that calls ``@register_evaluator``.
"""
from .base import (
    EvalResult,
    Evaluator,
    available_evaluators,
    get_evaluator,
    register_evaluator,
)

# Import built-ins for their registration side effects.
from .code_eval import CodeEvaluator
from .math_eval import MathEvaluator
from .science_eval import ScienceEvaluator

__all__ = [
    "EvalResult",
    "Evaluator",
    "available_evaluators",
    "get_evaluator",
    "register_evaluator",
    "CodeEvaluator",
    "MathEvaluator",
    "ScienceEvaluator",
]
