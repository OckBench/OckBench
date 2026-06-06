"""Math evaluator: regex extracts the answer block, the LLM judge scores it.

Extraction and scoring are separated. Regex is demoted to isolating the
``<answer>`` block; the LLM judge (required) decides correctness. There is no
regex-only scoring fallback. When a block is present the judge receives only the
extracted block; when absent it receives the full response so it can still try.
"""
import logging

from .base import EvalResult, Evaluator, register_evaluator
from .extraction import extract_answer_block
from .judge import Judge, build_judge

logger = logging.getLogger(__name__)


class MathEvaluator(Evaluator):
    def __init__(self, judge: Judge):
        self.judge = judge

    async def evaluate(self, problem, response: str) -> EvalResult:
        extracted, method = extract_answer_block(response)
        # Hand the isolated block to the judge when present; only fall back to
        # the whole response when no block was emitted.
        candidate = extracted if extracted is not None else response

        verdict = await self.judge.score(
            question=problem.problem,
            ground_truth=problem.answer,
            candidate=candidate,
        )
        return EvalResult(
            is_correct=verdict.correct,
            extracted_answer=(verdict.extracted_answer
                              if verdict.extracted_answer is not None else extracted),
            extraction_method=method,
            judge_reasoning=verdict.reasoning or verdict.error,
        )


def _missing_judge_fields(judge_cfg) -> list:
    """User-facing names of the judge fields that are unset/empty (after env+CLI
    resolution). All three are required to score math."""
    if judge_cfg is None:
        return ["judge model", "judge endpoint/base_url", "judge api-key"]
    missing = []
    if not judge_cfg.model:
        missing.append("judge model")
    if not judge_cfg.base_url:
        missing.append("judge endpoint/base_url")
    if not judge_cfg.api_key:
        missing.append("judge api-key")
    return missing


@register_evaluator("math")
def _build_math_evaluator(config) -> MathEvaluator:
    missing = _missing_judge_fields(config.judge)
    if missing:
        raise ValueError(
            "math scoring requires a fully configured LLM judge; missing: "
            + ", ".join(missing)
            + " (set via config.judge / --judge-model / --judge-base-url / --judge-api-key; "
            "the api-key also resolves from JUDGE_API_KEY or OPENAI_API_KEY). "
            "There is no regex-only scoring fallback for math."
        )
    return MathEvaluator(build_judge(config.judge))
