"""Pure scoring-core arithmetic shared by the runner and the cache aggregator.

This module is the single home for OckScore and the token-accounting roll-up.
Both the live runner and the cache->results aggregation call these functions so
the two can never diverge, and so the scoring core can be golden-tested in
isolation without constructing a runner or touching the network.

The arithmetic here is a behavior-preserving extraction of the formula that
previously lived inline in ``BenchmarkRunner._compute_summary``; it must not
change (the prompt/reasoning/answer/output split and the OckScore constant are a
scientific-validity contract).
"""
import math
from typing import List

from .schemas import EvaluationResult, ExperimentSummary

# OckScore token-penalty constants. Changing either breaks the published metric.
OCK_SCORE_TOKEN_WEIGHT = 10
OCK_SCORE_TOKEN_SCALE = 10000


def ock_score(accuracy: float, avg_tokens: float) -> float:
    """OckScore: accuracy (in percent) penalized by average token spend.

    ``accuracy - 10 * log(avg_tokens / 10000 + 1)`` — the exact published
    formula. ``accuracy`` is a percentage in [0, 100]; ``avg_tokens`` is the mean
    total tokens per problem.
    """
    return accuracy - OCK_SCORE_TOKEN_WEIGHT * math.log(avg_tokens / OCK_SCORE_TOKEN_SCALE + 1)


def summarize(results: List[EvaluationResult], duration: float) -> ExperimentSummary:
    """Aggregate per-problem results into an ``ExperimentSummary``.

    Pure function of the result list (plus wall-clock duration): the runner and
    the cache aggregator both call it, so a results file is always a faithful
    roll-up of its backing per-problem records.
    """
    total_problems = len(results)
    correct_count = sum(1 for r in results if r.correct)
    accuracy = (correct_count / total_problems * 100) if total_problems > 0 else 0

    total_prompt_tokens = sum(r.tokens.prompt_tokens for r in results)
    total_answer_tokens = sum(r.tokens.answer_tokens for r in results)
    total_reasoning_tokens = sum(r.tokens.reasoning_tokens for r in results)
    total_output_tokens = sum(r.tokens.output_tokens for r in results)
    total_tokens = sum(r.tokens.total_tokens for r in results)

    avg_tokens = total_tokens / total_problems if total_problems > 0 else 0
    avg_latency = sum(r.latency for r in results) / total_problems if total_problems > 0 else 0

    error_count = sum(1 for r in results if r.error or r.evaluator_error)
    score = ock_score(accuracy, avg_tokens)

    return ExperimentSummary(
        total_problems=total_problems,
        correct_count=correct_count,
        accuracy=accuracy,
        total_tokens=total_tokens,
        total_prompt_tokens=total_prompt_tokens,
        total_answer_tokens=total_answer_tokens,
        total_reasoning_tokens=total_reasoning_tokens,
        total_output_tokens=total_output_tokens,
        avg_tokens_per_problem=avg_tokens,
        avg_latency=avg_latency,
        total_duration=duration,
        error_count=error_count,
        ock_score=score,
    )
