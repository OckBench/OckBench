"""Evaluator interface + registry (the pluggable task/evaluator extension point).

Task type maps to an evaluator factory through one registry, parallel to the
provider registry. The runner resolves evaluators here, so there is no
hard-coded task->evaluator ``if/elif`` anywhere. Built-in evaluators register on
import (see ``src/evaluators``); an external task registers the same way — by
importing a module that calls ``@register_evaluator("name")`` — without editing
the runner or this file.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..core.schemas import BenchmarkConfig, Problem


@dataclass
class EvalResult:
    """Unified evaluation result from all evaluators."""
    is_correct: bool
    extracted_answer: Optional[Any]
    extraction_method: str
    judge_reasoning: Optional[str] = None
    tests_passed: Optional[int] = None
    tests_total: Optional[int] = None
    execution_error: Optional[str] = None


class Evaluator(ABC):
    """Scores one model response for one problem.

    Async because some scorers (the math LLM judge) make a network call; sync
    scorers simply return immediately.
    """

    @abstractmethod
    async def evaluate(self, problem: Problem, response: str) -> EvalResult:
        ...


# name -> factory(config) -> Evaluator. Populated by @register_evaluator.
_EVALUATOR_REGISTRY: Dict[str, Callable[[BenchmarkConfig], Evaluator]] = {}


def register_evaluator(name: str) -> Callable[[Callable], Callable]:
    """Decorator registering an evaluator factory ``(config) -> Evaluator``."""
    def _decorator(factory: Callable[[BenchmarkConfig], Evaluator]):
        if name in _EVALUATOR_REGISTRY:
            raise ValueError(f"evaluator '{name}' is already registered")
        _EVALUATOR_REGISTRY[name] = factory
        return factory

    return _decorator


def available_evaluators() -> List[str]:
    return sorted(_EVALUATOR_REGISTRY)


def get_evaluator(name: str, config: BenchmarkConfig) -> Evaluator:
    """Build the evaluator registered under ``name`` from ``config``.

    Raises ``ValueError`` enumerating the registered evaluators when ``name`` is
    not registered (fail fast, no silent default).
    """
    factory = _EVALUATOR_REGISTRY.get(name)
    if factory is None:
        raise ValueError(
            f"unknown evaluator '{name}'. Registered evaluators: "
            f"{', '.join(available_evaluators()) or '(none)'}"
        )
    return factory(config)
