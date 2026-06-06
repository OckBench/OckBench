"""AC-5: pluggable task/evaluator extension point (offline)."""
import asyncio
import inspect as _inspect

import pytest

import src.evaluators.base as ev_base
from src.core.schemas import BenchmarkConfig, Problem
from src.evaluators import EvalResult, Evaluator, available_evaluators, get_evaluator


def _config(evaluator_type):
    return BenchmarkConfig(dataset_path="d.jsonl", provider="gemini", model="m",
                           max_output_tokens=100, evaluator_type=evaluator_type)


def test_builtins_registered():
    for name in ("math", "science", "code"):
        assert name in available_evaluators()


def test_unknown_evaluator_enumerates_registered():
    with pytest.raises(ValueError) as exc:
        get_evaluator("nope", _config("nope"))
    msg = str(exc.value)
    assert "nope" in msg and "science" in msg


def test_external_task_evaluator_via_extension_point():
    name = "test-keyword-eval"
    try:
        @ev_base.register_evaluator(name)
        def _factory(config):
            class _KeywordEvaluator(Evaluator):
                async def evaluate(self, problem, response):
                    hit = str(problem.answer) in response
                    return EvalResult(is_correct=hit, extracted_answer=problem.answer,
                                      extraction_method="keyword")
            return _KeywordEvaluator()

        assert name in available_evaluators()
        # Selectable by name through the same dispatch the runner uses; no core edit.
        ev = get_evaluator(name, _config(name))
        result = asyncio.run(ev.evaluate(Problem(problem="q", answer="cat", id="1"),
                                         "the answer is cat"))
        assert result.is_correct is True and result.extraction_method == "keyword"
    finally:
        ev_base._EVALUATOR_REGISTRY.pop(name, None)


def test_dispatch_is_registry_based_not_ifelif():
    # get_evaluator must resolve via the registry, with no hard-coded task chain.
    src = _inspect.getsource(ev_base.get_evaluator)
    assert "_EVALUATOR_REGISTRY" in src
    assert "elif" not in src


def test_runner_uses_registry_get_evaluator():
    import src.core.runner as runner_mod
    src = _inspect.getsource(runner_mod.BenchmarkRunner.run)
    assert "get_evaluator" in src
